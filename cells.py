"""Grid cell model and classification (see docs/GAME_RULES.md).

Every grid cell is one of: EMPTY, BUCKET (with colour and visual face up/down), LOCK (a known
lock type such as ice or curtain, with its counter) or UNKNOWN. UNKNOWN cells are never tapped.

Lock types are data-driven: each one is an entry in ``CELL_LOCKS`` (single-cell locks, decided from
the cell's own features) or produced by a region detector in ``vision.py`` (multi-cell locks such
as curtains). Adding a new obstacle = adding an entry, not editing an if/else chain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from config import VisionConfig
from palette import Lab, Palette, bgr_to_lab, lab_distance


class Kind(str, Enum):
    EMPTY = "empty"
    BUCKET = "bucket"
    LOCK = "lock"
    UNKNOWN = "unknown"


@dataclass
class Cell:
    row: int
    col: int
    kind: Kind = Kind.UNKNOWN
    color: Optional[str] = None       # bucket colour (top stage for a tower)
    body: Optional[str] = None        # tower: colour of the stage below, else None
    face_up: Optional[bool] = None    # visual state of a bucket
    lock: Optional[str] = None        # lock type name (e.g. "ice", "curtain", "lock", "key")
    unlock: Optional[str] = None      # how it opens: "taps" | "fills" | "key" | "path"
    counter: Optional[int] = None     # number shown on the lock, if read
    region: Optional[int] = None      # id of the multi-cell lock this cell belongs to
    reason: str = ""                  # why it is UNKNOWN (for the log)
    lab: Optional[Lab] = None         # UNKNOWN because of its colour: the colour seen (LAB)
    occluded: bool = False            # hidden by a game hint arrow (hints.py): never tapped
    seen_lab: Optional[Lab] = None    # bucket: the LAB its top colour was read from (debug)
    seen_from: str = ""               # ... and where: "rim" (open bucket) | "top" | "disc"

    @property
    def is_obstacle(self) -> bool:
        """Blocks the path to the exit (everything except an empty cell)."""
        return self.kind is not Kind.EMPTY

    def label(self) -> str:
        """Same notation as dataset/annotations.json."""
        if self.kind is Kind.EMPTY:
            return "."
        if self.kind is Kind.LOCK and self.lock in ICON_LOCKS:
            return f"{self.lock}:{self.color or '?'}"
        if self.kind is Kind.LOCK:
            short = {"curtain": "cur"}.get(self.lock or "", self.lock or "lock")
            return f"{short}{'' if self.counter is None else self.counter}"
        if self.kind is Kind.BUCKET:
            col = f"{self.color}/{self.body}" if self.body else f"{self.color}"
            return f"{col}-{'up' if self.face_up else 'down'}"
        return "?"


@dataclass
class CellFeatures:
    empty_frac: float      # share of pixels with the empty-cell colour
    white_frac: float      # bright, unsaturated pixels (digits, highlights)
    dark_frac: float       # very dark pixels (digit / object outlines)
    face_ratio: float      # median V of the centre / rim brightness of the top stage
    top_lab: Optional[Lab]     # colour of the upper part of the object (top stage / rim)
    body_lab: Optional[Lab]    # colour of the lower part of the object
    hue_cyan_frac: float   # share of light-cyan pixels (ice body)
    centre_lab: Optional[Lab] = None  # upper middle: inside of an open bucket / a tower's disc
    ring_lab: Optional[Lab] = None    # around it: the rim of an open bucket / the dome under a disc
    rim_lab: Optional[Lab] = None     # brightest object pixels of the top stage: an open bucket's
                                      # rim (its inside is shaded darker: L183 055 r4c0 green)
    icon: Optional[str] = None        # icon template matched in the cell ("lock", "key"), if any
    icon_color: Optional[str] = None  # its colour from the palette's ``keys`` group
    icon_lab: Optional[Lab] = None    # the icon's colour as seen


@dataclass(frozen=True)
class CellLock:
    """A single-cell lock type: how to recognise it and how it opens."""
    name: str
    unlock: str                                            # "taps" | "fills"
    detect: Callable[[CellFeatures, VisionConfig], bool]
    counter_polarity: str = "light"                         # digit colour for digits.py
    coloured: bool = False         # has a colour (lock/key pairs); unknown colour = UNKNOWN cell
    before_empty: bool = False     # checked before "empty" (a key sits on an empty-looking cell)


def _is_ice(f: CellFeatures, v: VisionConfig) -> bool:
    return f.white_frac >= v.ice_min_white and f.dark_frac >= v.ice_min_dark and f.hue_cyan_frac > 0.3


# Icon locks (vision.IconMatcher): a LOCK covers buckets and opens when the KEY of the same
# colour flies to it; the key flies once the path rule connects its cell to the exit.
ICON_LOCKS = ("lock", "key")

# Single-cell locks, checked in order after "empty" (unless before_empty) and before "bucket".
CELL_LOCKS: List[CellLock] = [
    CellLock("lock", "key", lambda f, v: f.icon == "lock", coloured=True, before_empty=True),
    CellLock("key", "path", lambda f, v: f.icon == "key", coloured=True, before_empty=True),
    CellLock("ice", "taps", _is_ice),
]


def icon_lab(cell_bgr: np.ndarray) -> Optional[Lab]:
    """Median LAB of the saturated, lit pixels (the coloured body of a lock / key icon)."""
    import cv2

    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    mask = (hsv[..., 1] > 110) & (hsv[..., 2] > 120)
    if mask.sum() < 20:
        return None
    return tuple(float(x) for x in np.median(bgr_to_lab(cell_bgr)[mask], axis=0))  # type: ignore[return-value]


def _object_pixels(cell_bgr: np.ndarray, lab: np.ndarray, palette: Palette,
                   v: VisionConfig) -> np.ndarray:
    """Mask of pixels that belong to the object in the cell (not panel/empty background,
    not black outline)."""
    import cv2

    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    bg = lab_distance(lab, palette.ui["empty_cell"]) < v.empty_tol
    bg |= lab_distance(lab, palette.ui["panel_bg"]) < v.panel_tol
    return ~bg & (hsv[..., 2] > 60)


def _bright_lab(cell_bgr: np.ndarray, lab: np.ndarray, mask: np.ndarray) -> Optional[Lab]:
    """Median LAB of the brighter half of the masked pixels (rim / lit dome, not the shaded
    inside of a face-up bucket)."""
    if mask.sum() < 20:
        return None
    vals = cell_bgr.max(axis=2)[mask]
    sel = lab[mask][vals >= np.percentile(vals, 50)]
    return tuple(float(x) for x in np.median(sel, axis=0))  # type: ignore[return-value]


def cell_features(cell_bgr: np.ndarray, palette: Palette, v: VisionConfig) -> CellFeatures:
    """Features of one grid cell crop (the cell's own rectangle, slightly inset)."""
    import cv2

    palette.require_ui("empty_cell", "panel_bg")
    h, w = cell_bgr.shape[:2]
    lab = bgr_to_lab(cell_bgr)
    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
    S, V = hsv[..., 1], hsv[..., 2]
    empty = lab_distance(lab, palette.ui["empty_cell"]) < v.empty_tol
    cy, cx, ry, rx = h // 2, w // 2, max(1, h // 5), max(1, w // 5)
    centre = float(np.median(V[cy - ry:cy + ry, cx - rx:cx + rx]))
    obj = _object_pixels(cell_bgr, lab, palette, v)
    upper = obj.copy()
    upper[int(0.6 * h):] = False   # top stage only: lower tower stages and their rims are ignored
    # reference = the rim (bright percentile), not the median: a tall tower's top stage is drawn
    # higher, so its rim is mostly above the crop and the median would be the shaded inside
    ref = (float(np.percentile(V[upper], v.face_ref_percentile)) if upper.sum() >= 20
           else float(np.percentile(V, 90)))
    top, body = obj.copy(), obj.copy()
    top[h // 2:] = False
    body[: h // 2] = False
    mid = np.zeros_like(obj)   # where a tower's disc sits: the upper middle of the crop
    mid[int(0.1 * h):int(0.45 * h), cx - rx:cx + rx] = True
    mid &= obj
    ring = np.zeros_like(obj)  # around it, clear of a large disc: rim / dome
    ring[: int(0.55 * h), int(0.1 * w):int(0.9 * w)] = True
    ring[int(0.05 * h):int(0.5 * h), cx - int(1.6 * rx):cx + int(1.6 * rx)] = False
    ring &= obj
    med = lambda m: (tuple(float(x) for x in np.median(lab[m], axis=0))  # noqa: E731
                     if m.sum() >= 10 else None)
    rim = upper & (V >= np.percentile(V[upper], v.rim_percentile)) if upper.sum() >= 20 else upper
    return CellFeatures(
        empty_frac=float(empty.mean()),
        white_frac=float(((S < 50) & (V > 235)).mean()),
        dark_frac=float((V < 90).mean()),
        face_ratio=centre / max(1.0, ref),
        top_lab=_bright_lab(cell_bgr, lab, top),
        body_lab=_bright_lab(cell_bgr, lab, body),
        hue_cyan_frac=float(((hsv[..., 0] >= 90) & (hsv[..., 0] <= 105) & (S > 90)).mean()),
        centre_lab=med(mid),  # type: ignore[arg-type]
        ring_lab=med(ring),  # type: ignore[arg-type]
        rim_lab=med(rim),  # type: ignore[arg-type]
    )


def colour_distances(lab: Lab, palette: Palette) -> List[Tuple[float, str]]:
    """LAB distance to every bucket colour of the palette (nearest look), nearest first."""
    return sorted((min(float(np.linalg.norm(np.asarray(lab) - np.asarray(c))) for c in looks),
                   name) for name, looks in palette.buckets.items())


def _distinct(lab: Lab, best: str, other: str, palette: Palette, margin: float) -> bool:
    """True if ``lab`` is clearly closer to ``best`` than to ``other`` (a real second colour,
    not the shaded inside of a one-colour bucket)."""
    d = dict((n, x) for x, n in colour_distances(lab, palette))
    return d[other] - d[best] >= margin


def empty_fraction(region_bgr: np.ndarray, palette: Palette, v: VisionConfig) -> float:
    """Share of empty-cell coloured pixels. Callers pass the UPPER half of the plain cell
    rectangle: objects overhang upward only, so that part shows just this cell's own object."""
    return float((lab_distance(bgr_to_lab(region_bgr), palette.ui["empty_cell"])
                  < v.empty_tol).mean())


def classify_cell(row: int, col: int, f: CellFeatures, palette: Palette,
                  v: VisionConfig) -> Cell:
    """Classify a cell that is not covered by a multi-cell lock."""
    def lock_cell(lock: CellLock) -> Cell:
        if lock.coloured and f.icon_color is None:
            return Cell(row, col, Kind.UNKNOWN, reason=f"{lock.name} colour not in palette",
                        lab=f.icon_lab)
        return Cell(row, col, Kind.LOCK, lock=lock.name, unlock=lock.unlock,
                    color=f.icon_color if lock.coloured else None)

    for lock in CELL_LOCKS:
        if lock.before_empty and lock.detect(f, v):
            return lock_cell(lock)
    if f.empty_frac >= v.empty_min_frac:
        return Cell(row, col, Kind.EMPTY)
    for lock in CELL_LOCKS:
        if not lock.before_empty and lock.detect(f, v):
            return lock_cell(lock)
    if f.top_lab is None and f.body_lab is None:
        return Cell(row, col, Kind.UNKNOWN, reason="no object pixels")
    face_up = f.face_ratio < v.face_up_max_ratio
    # an open bucket's inside is shaded (light green reads dark green): its colour is the rim's
    top_lab, seen = ((f.rim_lab, "rim") if face_up and f.rim_lab is not None
                     else (f.top_lab or f.body_lab, "top"))
    top, dt = palette.nearest(top_lab, "buckets", v.color_max_dist)
    body, db = palette.nearest(f.body_lab or f.top_lab, "buckets", v.color_max_dist)
    if top is None and body is None:
        return Cell(row, col, Kind.UNKNOWN,
                    reason=f"colour not in palette (dist {min(dt, db):.1f})",
                    lab=top_lab)
    color = top or body
    # face-down tower seen from above: a disc of the top stage's colour on the body's dome, so
    # the ring around the centre is another colour. An open bucket's ring is its own rim, the
    # same colour as its inside (also for a face-up tower: L183 shot 054 r2c5).
    if f.centre_lab is not None and f.ring_lab is not None:
        disc, _ = palette.nearest(f.centre_lab, "buckets", v.color_max_dist)
        dome, _ = palette.nearest(f.ring_lab, "buckets", v.color_max_dist)
        apart = float(np.linalg.norm(np.subtract(f.centre_lab, f.ring_lab)))
        if disc is not None and dome is not None and disc != dome and apart >= v.disc_min_dist:
            return Cell(row, col, Kind.BUCKET, color=disc, body=dome, face_up=False,
                        seen_lab=f.centre_lab, seen_from="disc")
    tower = (body is not None and top is not None and body != top
             and _distinct(top_lab, top, body, palette, v.tower_min_margin))
    return Cell(row, col, Kind.BUCKET, color=color, body=body if tower else None,
                face_up=face_up, seen_lab=top_lab, seen_from=seen)


def path_playable(cells: Sequence[Sequence[Optional[Cell]]],
                  exits: Sequence[Tuple[int, int]]) -> List[List[bool]]:
    """Path rule: a bucket is playable iff a 4-neighbour path of EMPTY cells connects it to an
    exit cell (front-row cells adjacent to the conveyor exit). An exit cell itself counts as
    connected. ``cells[r][c]`` is None where the lattice has no grid cell."""
    conn = exit_connected(cells, exits)
    return [[conn[r][c] and cell is not None and cell.kind is Kind.BUCKET
             for c, cell in enumerate(row)] for r, row in enumerate(cells)]


def exit_connected(cells: Sequence[Sequence[Optional[Cell]]],
                   exits: Sequence[Tuple[int, int]]) -> List[List[bool]]:
    """Occupied cells (buckets, locks, unknown) that the path rule connects to the exit: an exit
    cell itself, or a cell next to an empty cell that an empty path links to the exit."""
    rows = len(cells)
    cols = len(cells[0]) if rows else 0
    reach = [[False] * cols for _ in range(rows)]   # empty cells connected to the exit
    queue = [(r, c) for r, c in exits if cells[r][c] is not None and not cells[r][c].is_obstacle]
    for r, c in queue:
        reach[r][c] = True
    while queue:
        r, c = queue.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < rows and 0 <= nc < cols and not reach[nr][nc]:
                n = cells[nr][nc]
                if n is not None and not n.is_obstacle:
                    reach[nr][nc] = True
                    queue.append((nr, nc))
    exit_set = set(map(tuple, exits))
    out = [[False] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            cell = cells[r][c]
            if cell is None or not cell.is_obstacle:
                continue
            nbrs = ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
            out[r][c] = (r, c) in exit_set or any(
                0 <= nr < rows and 0 <= nc < cols and reach[nr][nc] for nr, nc in nbrs)
    return out
