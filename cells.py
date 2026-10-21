"""Grid cell model and classification (see docs/GAME_RULES.md).

Every grid cell is one of: EMPTY, BUCKET (with colour and visual face up/down), LOCK (a known
lock type such as ice, curtain, chest or mystery bucket, with its counter) or UNKNOWN. UNKNOWN
cells are never tapped. A wall (L186) is a hole in the grid panel, drawn in the panel's own
colour: the lattice has no cell there (None, 'x'), which blocks paths like a wall.

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
    hidden: int = 0                   # lookahead (state.py): buckets of unknown colour here
    shell: bool = False               # L188 shell bucket: collects nothing until its shell drops in

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
            return f"{col}-{'up' if self.face_up else 'down'}{'+shell' if self.shell else ''}"
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
    wood_frac: float = 0.0            # share of chest-wood pixels (palette ui "chest_wood")
    grey_frac: float = 0.0            # share of mystery-grey pixels (palette ui "mystery_grey")
    split_bg: float = 0.0  # widest background (empty / panel) row across the middle: 2 objects
    shell_score: float = 0.0  # L188 shell icon on the bucket (icons.ShellMatcher)
    sat_lab: Optional[Lab] = None   # median of the saturated object pixels (no grey shell)


@dataclass(frozen=True)
class CellLock:
    """A single-cell lock type: how to recognise it and how it opens."""
    name: str
    unlock: str                                            # "taps" | "fills"
    detect: Callable[[CellFeatures, VisionConfig], bool]
    counter_polarity: str = "light"                         # digit colour for digits.py
    coloured: bool = False         # has a colour (lock/key pairs); unknown colour = UNKNOWN cell
    before_empty: bool = False     # checked before "empty" (a key sits on an empty-looking cell)
    counter: str = "optional"      # "optional" | "required" (unread: not this lock) | "none"


def _is_ice(f: CellFeatures, v: VisionConfig) -> bool:
    return f.white_frac >= v.ice_min_white and f.dark_frac >= v.ice_min_dark and f.hue_cyan_frac > 0.3


# Icon locks (vision.IconMatcher): a LOCK covers buckets and opens when the KEY of the same
# colour flies to it; the key flies once the path rule connects its cell to the exit.
# chain (L189, user 2026-10-21): a padlock with chains - a GATE, impassable for the path rule
# until the key of its colour flies to it; then its cells are empty (nothing under it).
ICON_LOCKS = ("lock", "key", "chain")

# Single-cell locks, checked in order after "empty" (unless before_empty) and before "bucket".
# chest (L186): a wooden box with a number N; opens when the path rule connects it (like a key)
# and scatters N buckets into free cells. mystery (L186): a grey bucket with "?", its colour is
# revealed when a 4-neighbour bucket is tapped (the "?" reads as a digit: no counter).
CELL_LOCKS: List[CellLock] = [
    CellLock("lock", "key", lambda f, v: f.icon == "lock", coloured=True, before_empty=True),
    CellLock("key", "path", lambda f, v: f.icon == "key", coloured=True, before_empty=True),
    CellLock("chain", "key", lambda f, v: f.icon == "chain", coloured=True, before_empty=True),
    # ice (user): always shows its number N; a cyan bucket (L189 r4c6) passes the colour test
    CellLock("ice", "taps", _is_ice, counter="required"),
    CellLock("chest", "path", lambda f, v: f.wood_frac >= v.chest_min_wood, counter="required"),
    CellLock("mystery", "adjacent", lambda f, v: f.grey_frac >= v.mystery_min_grey,
             counter="none"),
]
CELL_LOCK_BY_NAME = {lock.name: lock for lock in CELL_LOCKS}


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


def _split_band(obj: np.ndarray, V: np.ndarray, h: int, w: int) -> float:
    """Two separate objects stacked in one cell (e.g. two buckets of a lane in a cell of a
    too-coarse lattice, run 20261020-122638 "cream/blue-up"): a background row through the
    middle (centre 40 % of the width, 25-80 % of the height) with object pixels both above and
    below it. Score = min(background share of that row, object share above, object share
    below), best row. One object that fills only part of the cell (a fading padlock, a bucket
    flying off) has nothing below the row: 0."""
    cols = slice(int(0.3 * w), int(0.7 * w))
    bg = (~obj & (V > 60))[:, cols].mean(axis=1)
    ob = obj[:, cols].mean(axis=1)
    best = 0.0
    for r in range(int(0.25 * h), int(0.8 * h)):
        above, below = ob[:r], ob[r + 1:]
        if above.size and below.size:
            best = max(best, min(float(bg[r]), float(above.mean()), float(below.mean())))
    return best


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
    split_bg = _split_band(obj, V, h, w)

    def share(name: str, tol: float) -> float:
        if name not in palette.ui:
            return 0.0
        return float((lab_distance(lab, palette.ui[name]) < tol).mean())

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
        wood_frac=share("chest_wood", v.chest_tol),
        grey_frac=share("mystery_grey", v.mystery_tol),
        split_bg=split_bg,
        sat_lab=med(obj & (S > 55) & (V > 100)),  # type: ignore[arg-type]
    )


def foot_lab(strip_bgr: np.ndarray, palette: Palette, v: VisionConfig) -> Optional[Lab]:
    """Median LAB of the object pixels in the strip under a cell's crop, if they fill at least
    ``foot_min_obj`` of it (else background / outline: no stage there)."""
    if strip_bgr.size == 0:
        return None
    lab = bgr_to_lab(strip_bgr)
    obj = _object_pixels(strip_bgr, lab, palette, v)
    if obj.mean() < v.foot_min_obj:
        return None
    return tuple(float(x) for x in np.median(lab[obj], axis=0))  # type: ignore[return-value]


def foot_stage(cell: Cell, lab: Optional[Lab], below: Optional[Cell], palette: Palette,
               v: VisionConfig) -> Optional[str]:
    """Lower stage of an open tower in a tall lane (run 20261020-122638 r1c0 orange on cream):
    the top stage fills the crop and the lower one shows only in the strip under it (``lab``,
    see foot_lab). None unless ``cell`` is a face-up one-colour bucket and the strip's colour is
    another bucket colour that the cell below cannot explain: a tall lock below (ice, curtain,
    chest, mystery: no colour) or an unknown cell overhangs into the strip (shots 004 ice,
    055 curtain), and so does an object of the same colour."""
    if (lab is None or cell.kind is not Kind.BUCKET or not cell.face_up or cell.body is not None
            or cell.color is None or cell.shell):   # a shell bucket is one bucket, never a tower
        return None
    if below is not None and (below.kind is Kind.UNKNOWN
                              or (below.kind is Kind.LOCK and below.color is None)):
        return None
    foot, _ = palette.nearest(lab, "buckets", v.color_max_dist)
    if (foot is None or foot == cell.color or (below is not None and below.color == foot)
            or not _distinct(lab, foot, cell.color, palette, v.tower_min_margin)):
        return None
    return foot


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
                  v: VisionConfig, skip: Sequence[str] = ()) -> Cell:
    """Classify a cell that is not covered by a multi-cell lock. ``skip``: lock types ruled out
    (a "required" counter that did not read: e.g. a yellow padlock fading over a bucket is no
    chest, a cyan bucket is no ice)."""
    def lock_cell(lock: CellLock) -> Cell:
        if lock.coloured and f.icon_color is None and lock.name != "chain":
            # a chain cell may show only chain links: its gate's padlock gives the colour
            # (state._merge_lock_icons)
            return Cell(row, col, Kind.UNKNOWN, reason=f"{lock.name} colour not in palette",
                        lab=f.icon_lab)
        return Cell(row, col, Kind.LOCK, lock=lock.name, unlock=lock.unlock,
                    color=f.icon_color if lock.coloured else None)

    locks = [lock for lock in CELL_LOCKS if lock.name not in skip]
    for lock in locks:
        if lock.before_empty and lock.detect(f, v):
            return lock_cell(lock)
    if f.empty_frac >= v.empty_min_frac:
        return Cell(row, col, Kind.EMPTY)
    if f.shell_score >= v.shell_threshold:   # before mystery: the grey shell is mystery-grey
        shell = _shell_bucket(row, col, f, palette, v)
        if shell is not None:
            return shell
    for lock in locks:
        if not lock.before_empty and lock.detect(f, v):
            return lock_cell(lock)
    if f.top_lab is None and f.body_lab is None:
        return Cell(row, col, Kind.UNKNOWN, reason="no object pixels")
    if f.split_bg >= v.split_bg_max:   # never one bucket / tower: two objects (misread lattice?)
        return Cell(row, col, Kind.UNKNOWN,
                    reason=f"two objects in one cell (background band {f.split_bg:.2f})")
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


def _shell_bucket(row: int, col: int, f: CellFeatures, palette: Palette,
                  v: VisionConfig) -> Optional[Cell]:
    """L188 shell bucket (user, 2026-10-21): face-up shows a GREY shell inside the open bucket,
    face-down a shell of the bucket's own colour on its dome (before: the grey shell read as
    mystery / a cream stage). Face-up = the shell (cell centre) is grey (chroma below
    ``vision.shell_grey_chroma``); face-down = the shell has the dome's colour. Colour from the
    saturated pixels. Anything else that matched the shell shape is a tower top (a disc of
    ANOTHER colour on a dome): None."""
    if f.centre_lab is None:
        return None
    grey = float(np.hypot(f.centre_lab[1], f.centre_lab[2])) < v.shell_grey_chroma
    # the saturated pixels: rim / dome / body, never the grey shell (an open darkred shell
    # bucket's lower half is mostly shell and read cream)
    lab = f.sat_lab or f.body_lab
    color, _ = palette.nearest(lab, "buckets", v.color_max_dist)
    if color is None:
        return None
    if not grey and palette.nearest(f.centre_lab, "buckets", v.color_max_dist)[0] != color:
        return None
    return Cell(row, col, Kind.BUCKET, color=color, face_up=grey, seen_lab=lab,
                seen_from="shell", shell=True)


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
