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
    lock: Optional[str] = None        # lock type name (e.g. "ice", "curtain")
    unlock: Optional[str] = None      # how the lock opens: "taps" | "fills"
    counter: Optional[int] = None     # number shown on the lock, if read
    region: Optional[int] = None      # id of the multi-cell lock this cell belongs to
    reason: str = ""                  # why it is UNKNOWN (for the log)

    @property
    def is_obstacle(self) -> bool:
        """Blocks the path to the exit (everything except an empty cell)."""
        return self.kind is not Kind.EMPTY

    def label(self) -> str:
        """Same notation as dataset/annotations.json."""
        if self.kind is Kind.EMPTY:
            return "."
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
    face_ratio: float      # median V of the centre / median V of the object's top part
    top_lab: Optional[Lab]     # colour of the upper part of the object (top stage / rim)
    body_lab: Optional[Lab]    # colour of the lower part of the object
    hue_cyan_frac: float   # share of light-cyan pixels (ice body)


@dataclass(frozen=True)
class CellLock:
    """A single-cell lock type: how to recognise it and how it opens."""
    name: str
    unlock: str                                            # "taps" | "fills"
    detect: Callable[[CellFeatures, VisionConfig], bool]
    counter_polarity: str = "light"                         # digit colour for digits.py


def _is_ice(f: CellFeatures, v: VisionConfig) -> bool:
    return f.white_frac >= v.ice_min_white and f.dark_frac >= v.ice_min_dark and f.hue_cyan_frac > 0.3


# Single-cell locks, checked in order after "empty" and before "bucket".
CELL_LOCKS: List[CellLock] = [
    CellLock("ice", "taps", _is_ice),
]


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
    upper[int(0.6 * h):] = False   # top stage only (a tower's lower stage is brighter)
    ref = float(np.median(V[upper])) if upper.sum() >= 20 else float(np.percentile(V, 90))
    top, body = obj.copy(), obj.copy()
    top[h // 2:] = False
    body[: h // 2] = False
    return CellFeatures(
        empty_frac=float(empty.mean()),
        white_frac=float(((S < 50) & (V > 235)).mean()),
        dark_frac=float((V < 90).mean()),
        face_ratio=centre / max(1.0, ref),
        top_lab=_bright_lab(cell_bgr, lab, top),
        body_lab=_bright_lab(cell_bgr, lab, body),
        hue_cyan_frac=float(((hsv[..., 0] >= 90) & (hsv[..., 0] <= 105) & (S > 90)).mean()),
    )


def _distinct(lab: Lab, best: str, other: str, palette: Palette, margin: float) -> bool:
    """True if ``lab`` is clearly closer to ``best`` than to ``other`` (a real second colour,
    not the shaded inside of a one-colour bucket)."""
    d = lambda name: min(float(np.linalg.norm(np.asarray(lab) - np.asarray(c)))  # noqa: E731
                         for c in palette.buckets[name])
    return d(other) - d(best) >= margin


def empty_fraction(region_bgr: np.ndarray, palette: Palette, v: VisionConfig) -> float:
    """Share of empty-cell coloured pixels. Callers pass the UPPER half of the plain cell
    rectangle: objects overhang upward only, so that part shows just this cell's own object."""
    return float((lab_distance(bgr_to_lab(region_bgr), palette.ui["empty_cell"])
                  < v.empty_tol).mean())


def classify_cell(row: int, col: int, f: CellFeatures, palette: Palette,
                  v: VisionConfig) -> Cell:
    """Classify a cell that is not covered by a multi-cell lock."""
    if f.empty_frac >= v.empty_min_frac:
        return Cell(row, col, Kind.EMPTY)
    for lock in CELL_LOCKS:
        if lock.detect(f, v):
            return Cell(row, col, Kind.LOCK, lock=lock.name, unlock=lock.unlock)
    if f.top_lab is None and f.body_lab is None:
        return Cell(row, col, Kind.UNKNOWN, reason="no object pixels")
    top, dt = palette.nearest(f.top_lab or f.body_lab, "buckets", v.color_max_dist)
    body, db = palette.nearest(f.body_lab or f.top_lab, "buckets", v.color_max_dist)
    if top is None and body is None:
        return Cell(row, col, Kind.UNKNOWN,
                    reason=f"colour not in palette (dist {min(dt, db):.1f})")
    color = top or body
    tower = (body is not None and top is not None and body != top
             and _distinct(f.top_lab, top, body, palette, v.tower_min_margin))
    return Cell(row, col, Kind.BUCKET, color=color, body=body if tower else None,
                face_up=f.face_ratio < v.face_up_max_ratio)


def path_playable(cells: Sequence[Sequence[Optional[Cell]]],
                  exits: Sequence[Tuple[int, int]]) -> List[List[bool]]:
    """Path rule: a bucket is playable iff a 4-neighbour path of EMPTY cells connects it to an
    exit cell (front-row cells adjacent to the conveyor exit). An exit cell itself counts as
    connected. ``cells[r][c]`` is None where the lattice has no grid cell."""
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
            if cell is None or cell.kind is not Kind.BUCKET:
                continue
            nbrs = ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
            out[r][c] = (r, c) in exit_set or any(
                0 <= nr < rows and 0 <= nc < cols and reach[nr][nc] for nr, nc in nbrs)
    return out
