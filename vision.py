"""Read the gameplay state from a frame of the iPhone screen (BGR, any resolution).

Everything is located relative to the screen (zones in config.yaml ``vision``) or detected
(grid lattice, curtains); colours come from the dataset palette (palette.py). Output positions
are relative 0..1 coordinates, ready for actions.Tapper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from cells import (Cell, Kind, cell_features, classify_cell, colour_distances, empty_fraction,
                   icon_lab, path_playable)
from config import VisionConfig
from digits import DigitReader
from hints import CellMemory, HintArrows, RecentUnlocks, clean, occlusion_mask
from belt import Slot, read_conveyor
from icons import IconMatcher
from palette import Palette, bgr_to_lab, lab_distance
from sand import read_sand
from unknown_colours import UnknownColour, unknown_sand
from zones import crop_rel  # noqa: F401  (re-exported)

Grid = List[List[Optional[Cell]]]   # None = no grid cell at this lattice position


@dataclass
class Lattice:
    x0: float          # pixels, top-left of cell (0, 0)
    y0: float
    cw: float          # cell size in pixels
    ch: float
    mask: np.ndarray   # rows x cols bool: lattice position is a grid cell

    @property
    def rows(self) -> int:
        return int(self.mask.shape[0])

    @property
    def cols(self) -> int:
        return int(self.mask.shape[1])

    def rect(self, r: int, c: int) -> Tuple[float, float, float, float]:
        x, y = self.x0 + c * self.cw, self.y0 + r * self.ch
        return x, y, x + self.cw, y + self.ch

    def exits(self) -> List[Tuple[int, int]]:
        """Front-row cells, adjacent to the conveyor exit at the top of the grid."""
        return [(0, c) for c in range(self.cols) if self.mask[0, c]]


class VisionError(RuntimeError):
    pass


def find_lattice(img: np.ndarray, palette: Palette, v: VisionConfig,
                 occluded: Optional[np.ndarray] = None) -> Lattice:
    """Find the grid panel (the hole in the purple background below the conveyor) and its
    cell lattice. Column pitch comes from the periodic cell edges; row pitch from the fixed
    cell aspect ratio. ``occluded``: frame mask of hint arrows (hints.py), never grid - they
    would join the grid to the booster buttons below it."""
    import cv2

    palette.require_ui("panel_bg")
    h, w = img.shape[:2]
    zx0, zy0 = int(v.grid_zone[0] * w), int(v.grid_zone[1] * h)
    zone = crop_rel(img, v.grid_zone)
    panel = lab_distance(bgr_to_lab(zone), palette.ui["panel_bg"]) < v.panel_tol
    # side strips start ``edge_margin`` inside the frame: a live crop can carry a few px of
    # window edge / chrome, which used to make every row fail the panel test
    edge, sw = int(v.edge_margin * w), max(2, int(0.05 * w))
    side = ((panel[:, edge:edge + sw].mean(axis=1) > 0.9)
            | (panel[:, w - edge - sw:w - edge].mean(axis=1) > 0.9))
    if not side.any():
        raise VisionError("grid panel background not found")
    ytop = int(np.argmax(side))
    grid = ~panel
    grid[:ytop] = False
    if occluded is not None:
        grid[occluded[zy0:zy0 + grid.shape[0], zx0:zx0 + grid.shape[1]]] = False
    if edge:
        grid[:, :edge] = grid[:, w - edge:] = False   # border strips are never the grid
    n, lab, st, _ = cv2.connectedComponentsWithStats(grid.astype(np.uint8), 4)
    cands = [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] > 0.002 * w * h]
    if not cands:
        raise VisionError("no grid region below the conveyor")
    k = min(cands, key=lambda i: st[i, cv2.CC_STAT_TOP])   # the grid touches the frame top
    g = lab == k
    ys, xs = np.nonzero(g)
    gx0, gx1, gy0, gy1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY).astype(np.float32)
    edges = np.abs(np.diff(gray, axis=1))
    edges[~g[:, 1:]] = 0
    prof = edges.sum(axis=0)
    prof -= prof.mean()
    ac = np.correlate(prof, prof, "full")[len(prof) - 1:]
    lo, hi = int(0.06 * w), int(0.2 * w)
    pitch = lo + int(np.argmax(ac[lo:hi]))
    cols = max(1, round((gx1 - gx0) / pitch))
    cw = (gx1 - gx0) / cols
    rows = max(1, round((gy1 - gy0) / (cw * v.cell_aspect)))
    ch = (gy1 - gy0) / rows
    occ = (np.zeros_like(g) if occluded is None
           else occluded[zy0:zy0 + g.shape[0], zx0:zx0 + g.shape[1]])
    mask = np.zeros((rows, cols), bool)
    for r in range(rows):
        for c in range(cols):
            win = (slice(int(gy0 + (r + .25) * ch), int(gy0 + (r + .75) * ch)),
                   slice(int(gx0 + (c + .25) * cw), int(gx0 + (c + .75) * cw)))
            a, seen = g[win], ~occ[win]
            if a.size and seen.mean() < 0.25:   # mostly under a hint arrow: keep it (occluded)
                mask[r, c] = True
            else:
                mask[r, c] = a.size > 0 and a[seen].mean() > 0.5
    return Lattice(zx0 + float(gx0), zy0 + float(gy0), float(cw), float(ch), mask)


@dataclass
class Region:
    """A multi-cell lock (e.g. a curtain) and the cells it covers."""
    lock: str
    unlock: str
    box: Tuple[int, int, int, int]          # pixels x0, y0, x1, y1
    cells: List[Tuple[int, int]]
    counter: Optional[int] = None


def find_curtains(img: np.ndarray, lat: Lattice, palette: Palette, v: VisionConfig,
                  reader: Optional[DigitReader] = None) -> List[Region]:
    """Curtains: a continuous gold valance bar wider than 1.5 cells, with gold side posts going
    down. The extent comes from the frame on screen, so any size (2x2, 2x3, ...) works."""
    palette.require_ui("curtain_gold")
    x0, y0 = int(lat.x0), int(lat.y0)
    x1, y1 = int(lat.x0 + lat.cols * lat.cw), int(lat.y0 + lat.rows * lat.ch)
    area = img[y0:y1, x0:x1]
    gold = lab_distance(bgr_to_lab(area), palette.ui["curtain_gold"]) < 18
    min_run = int(1.5 * lat.cw)
    max_gap = max(2, int(0.15 * lat.ch))   # posts are continuous; a longer gap = next curtain
    regions: List[Region] = []
    used = np.zeros(gold.shape, bool)
    for y in range(gold.shape[0]):
        row = np.concatenate([[0], (gold[y] & ~used[y]).astype(np.int8), [0]])
        d = np.diff(row)
        starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
        runs = [(s, e) for s, e in zip(starts, ends) if e - s >= min_run]
        for s, e in runs:
            post = max(2, (e - s) // 25)
            sides = gold[y:, s:s + post].any(axis=1) | gold[y:, e - post:e].any(axis=1)
            bottom = y + _run_end(sides, max_gap)
            if bottom - y < 0.8 * lat.ch:
                continue
            used[y:bottom, s:e] = True
            box = (x0 + int(s), y0 + y, x0 + int(e), y0 + bottom)
            covered = [(r, c) for r in range(lat.rows) for c in range(lat.cols)
                       if lat.mask[r, c] and box[0] <= lat.x0 + (c + .5) * lat.cw <= box[2]
                       and box[1] <= lat.y0 + (r + .5) * lat.ch <= box[3]]
            counter = None
            if reader is not None and covered:
                bx0, by0, bx1, by1 = box
                sign = img[by0 + (by1 - by0) * 3 // 10: by0 + (by1 - by0) * 7 // 10,
                           bx0 + (bx1 - bx0) // 4: bx1 - (bx1 - bx0) // 4]
                counter = reader.read_int(sign, "light")
            regions.append(Region("curtain", "fills", box, covered, counter))
    return regions


def _run_end(on: np.ndarray, max_gap: int) -> int:
    """End (exclusive) of the run of True starting at index 0, bridging gaps <= max_gap."""
    end, gap = 0, 0
    for i, v in enumerate(on):
        if v:
            end, gap = i + 1, 0
        else:
            gap += 1
            if gap > max_gap:
                break
    return end


def cell_box(lat: Lattice, r: int, c: int, v: VisionConfig) -> Tuple[int, int, int, int]:
    """The cell's rectangle shifted up by ``cell_shift_up`` (objects are drawn overhanging into
    the row above) and inset by ``cell_inset`` on each side: (x0, y0, x1, y1) pixels."""
    x0, y0, x1, y1 = lat.rect(r, c)
    dy = v.cell_shift_up * lat.ch
    ix, iy = v.cell_inset * lat.cw, v.cell_inset * lat.ch
    return int(x0 + ix), max(0, int(y0 - dy + iy)), int(x1 - ix), int(y1 - dy - iy)


def cell_crop(img: np.ndarray, lat: Lattice, r: int, c: int, v: VisionConfig) -> np.ndarray:
    x0, y0, x1, y1 = cell_box(lat, r, c, v)
    return img[y0:y1, x0:x1]


def lock_digits_roi(img: np.ndarray, lat: Lattice, r: int, c: int, v: VisionConfig) -> np.ndarray:
    """Where a single-cell lock shows its counter: the cell shifted up like the objects."""
    x0, y0, x1, y1 = lat.rect(r, c)
    dy = v.cell_shift_up * lat.ch
    return img[max(0, int(y0 - dy)):int(y1 - dy / 2), int(x0):int(x1)]


@dataclass
class GameView:
    lattice: Lattice
    grid: Grid
    regions: List[Region]
    playable: List[List[bool]]                 # path rule (BFS)
    disagreements: List[Tuple[int, int]]       # visual face state != path rule
    counter: Optional[int] = None              # k of "k/5"
    slots: List[Slot] = field(default_factory=list)
    sand: Dict[str, float] = field(default_factory=dict)     # share of picture area per colour
    exposed: Dict[str, float] = field(default_factory=dict)  # share of the bottom band
    lower: Dict[str, float] = field(default_factory=dict)    # share of the picture's lower part
    frame_size: Tuple[int, int] = (0, 0)       # w, h in pixels
    unknown_colours: List[UnknownColour] = field(default_factory=list)
    hints: List[Tuple[int, int, int, int]] = field(default_factory=list)  # hint arrow boxes
    colour_notes: List[str] = field(default_factory=list)   # close colour calls (debug)
    unlocked: Set[Tuple[int, int]] = field(default_factory=set)  # key lock just opened

    def labels(self) -> List[str]:
        return [" ".join("x" if c is None else c.label() for c in row) for row in self.grid]

    def tappable(self) -> List[Tuple[int, int]]:
        """Buckets that are playable by the path rule AND look face-up (both must agree; in a
        cell whose key lock just opened the padlock animation hides the face: path rule alone)."""
        return [(r, c) for r, row in enumerate(self.grid) for c, cell in enumerate(row)
                if cell is not None and cell.kind is Kind.BUCKET and self.playable[r][c]
                and (cell.face_up or (r, c) in self.unlocked) and not cell.occluded]

    def cell_center_rel(self, r: int, c: int) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.lattice.rect(r, c)
        w, h = self.frame_size
        return (x0 + x1) / 2 / w, (y0 + y1) / 2 / h

    def unknown_cells(self) -> List[Cell]:
        """UNKNOWN cells (not the ones hidden by a hint arrow: those are not a new mechanic)."""
        return [c for row in self.grid for c in row
                if c is not None and c.kind is Kind.UNKNOWN and not c.occluded]


def read_grid(img: np.ndarray, palette: Palette, v: VisionConfig,
              reader: Optional[DigitReader] = None,
              icons: Optional[IconMatcher] = None, occluded: Optional[np.ndarray] = None,
              memory: Optional[CellMemory] = None) -> Tuple[Lattice, Grid, List[Region]]:
    """``occluded``: hint-arrow mask (``img`` already inpainted there). A cell partly under it is
    read through the inpainted pixels; if more than ``hint_cover_max`` of its crop is hidden, or
    it reads UNKNOWN through the mask, it is occluded: its last clean reading from ``memory`` if
    there is one, else UNKNOWN; either way never tapped and never saved as unknown."""
    lat = find_lattice(img, palette, v, occluded)
    regions = find_curtains(img, lat, palette, v, reader)
    grid: Grid = [[None] * lat.cols for _ in range(lat.rows)]
    for i, reg in enumerate(regions):
        for r, c in reg.cells:
            grid[r][c] = Cell(r, c, Kind.LOCK, lock=reg.lock, unlock=reg.unlock,
                              counter=reg.counter, region=i)
    for r in range(lat.rows):
        for c in range(lat.cols):
            if not lat.mask[r, c] or grid[r][c] is not None:
                continue
            crop = cell_crop(img, lat, r, c, v)
            f = cell_features(crop, palette, v)
            x0, y0, x1, y1 = lat.rect(r, c)
            ix = v.cell_inset * lat.cw
            upper = img[int(y0 + v.cell_inset * lat.ch):int((y0 + y1) / 2), int(x0 + ix):int(x1 - ix)]
            f.empty_frac = empty_fraction(upper, palette, v)
            if icons is not None:
                kind, score = icons.match(crop)
                if kind is not None and score >= v.icon_threshold:
                    f.icon = kind
                    f.icon_lab = icon_lab(crop)
                    f.icon_color = (None if f.icon_lab is None else
                                    palette.nearest(f.icon_lab, "keys", v.color_max_dist)[0])
            cell = classify_cell(r, c, f, palette, v)
            if cell.kind is Kind.LOCK and reader is not None:
                cell.counter = reader.read_int(lock_digits_roi(img, lat, r, c, v), "light")
            if occluded is not None:
                bx0, by0, bx1, by1 = cell_box(lat, r, c, v)
                hidden = occluded[by0:by1, bx0:bx1]
                cover = float(hidden.mean()) if hidden.size else 0.0
                if cover > v.hint_cover_max or (cover > 0.02 and cell.kind is Kind.UNKNOWN):
                    cell = (memory.recall(lat, (r, c)) if memory is not None else None) or Cell(
                        r, c, Kind.UNKNOWN, reason="hidden by hint arrow, no earlier reading")
                    cell.occluded = True
            grid[r][c] = cell
    return lat, grid, regions


def read_gameplay(img: np.ndarray, palette: Palette, v: VisionConfig,
                  reader: Optional[DigitReader] = None,
                  log: Callable[[str], None] = print,
                  icons: Optional[IconMatcher] = None, hints: Optional[HintArrows] = None,
                  memory: Optional[CellMemory] = None,
                  unlocks: Optional[RecentUnlocks] = None, now: float = 0.0) -> GameView:
    """``hints``: game hint arrows to ignore (hints.py); ``memory``: last clean cell readings,
    used for cells the arrows hide and updated with this frame's clean cells; ``unlocks``: cells
    whose key lock opened recently (``now`` = frame time in s), where the path rule alone decides
    (their face reads down under the padlock animation)."""
    arrows = hints.find(img, v) if hints is not None else []
    occ = occlusion_mask(img.shape, arrows) if arrows else None
    if arrows:
        log(f"hint arrows: {len(arrows)} masked out")
    lat, grid, regions = read_grid(img if occ is None else clean(img, occ), palette, v, reader,
                                   icons, occ, memory)
    if memory is not None:
        memory.store(lat, grid)
    playable = path_playable(grid, lat.exits())
    unlocked = unlocks.update(lat, grid, now) if unlocks is not None else set()
    disagree = []
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if (cell is not None and cell.kind is Kind.BUCKET and not cell.occluded
                    and cell.face_up != playable[r][c] and (r, c) not in unlocked):
                disagree.append((r, c))
                log(f"warning: cell r{r}c{c} ({cell.label()}) looks "
                    f"{'face-up' if cell.face_up else 'face-down'} but the path rule says "
                    f"{'playable' if playable[r][c] else 'closed'} — not tapping it")
    k, slots = read_conveyor(img, palette, v, reader)
    sand, exposed, lower = read_sand(img, palette, v) if palette.sand else ({}, {}, {})
    unknown = [UnknownColour(cell.reason.split()[0] if cell.reason.startswith(("lock", "key"))
                             else "bucket", cell.lab, cell=(cell.row, cell.col))
               for row in grid for cell in row
               if cell is not None and cell.kind is Kind.UNKNOWN and cell.lab is not None
               and not cell.occluded]
    unknown += unknown_sand(img, palette, v)
    for u in unknown:
        log(f"warning: {u.describe()} — not in the palette (dataset/unknown_colours/, "
            f"python -m tools.add_colour)")
    return GameView(lat, grid, regions, playable, disagree, k, slots, sand, exposed, lower,
                    (img.shape[1], img.shape[0]), unknown, [b for b, _ in arrows],
                    colour_notes(grid, palette, v), unlocked)


def colour_notes(grid: Grid, palette: Palette, v: VisionConfig) -> List[str]:
    """Bucket cells whose runner-up colour is also within ``color_max_dist``, and every bucket
    of a ``colour_watch`` pair (e.g. green / darkgreen) with its distance to both:
    "r4c0 green-up (rim): green 2.4, darkgreen 21.1"; "CLOSE" when the two are less than
    ``tower_min_margin`` apart."""
    out = []
    for row in grid:
        for cell in row:
            if cell is None or cell.kind is not Kind.BUCKET or cell.seen_lab is None:
                continue
            dist = colour_distances(cell.seen_lab, palette)
            (d1, n1), (d2, n2) = dist[:2]
            pair = next((p for p in v.colour_watch if cell.color in p), None)
            if pair is not None:   # always both colours of a watched close pair
                d = {n: x for x, n in dist}
                (d1, n1), (d2, n2) = sorted((d.get(n, float("inf")), n) for n in pair)
            if pair is not None or d2 <= v.color_max_dist:
                out.append(f"r{cell.row}c{cell.col} {cell.label()} ({cell.seen_from}): "
                           f"{n1} {d1:.1f}, {n2} {d2:.1f}"
                           + (" CLOSE" if d2 - d1 < v.tower_min_margin else ""))
    return out


def draw(img: np.ndarray, gv: GameView) -> np.ndarray:
    """Debug image: green = tappable, red = closed/locked/unknown, yellow = visual state and
    path rule disagree, grey = empty, magenta = curtain extent."""
    import cv2

    out = img.copy()
    for r, row in enumerate(gv.grid):
        for c, cell in enumerate(row):
            if cell is None:
                continue
            x0, y0, x1, y1 = (int(round(a)) for a in gv.lattice.rect(r, c))
            color, width = ((0, 200, 0), 2) if (r, c) in gv.tappable() else ((0, 0, 220), 2)
            if (r, c) in gv.disagreements:
                color = (0, 220, 220)
            if cell.kind is Kind.EMPTY:
                color, width = (170, 170, 170), 1
            cv2.rectangle(out, (x0, y0), (x1, y1), color, width)
            cv2.putText(out, cell.label(), (x0 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(out, cell.label(), (x0 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (255, 255, 255), 1, cv2.LINE_AA)
    for reg in gv.regions:
        cv2.rectangle(out, reg.box[:2], reg.box[2:], (255, 0, 255), 2)
    for x0, y0, x1, y1 in gv.hints:   # hint arrows (ignored): dashed-looking grey box
        cv2.rectangle(out, (x0, y0), (x1, y1), (128, 128, 128), 1)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    import cv2

    from config import load_config

    ap = argparse.ArgumentParser(prog="python -m vision",
                                 description="Read the gameplay state from screenshots.")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    pal = Palette.load(cfg.paths.palette)
    reader = DigitReader.load(Path(cfg.paths.dataset) / "digits.npz")
    icons = IconMatcher.load(Path(cfg.paths.dataset) / "templates" / "icons")
    hints = HintArrows.load(Path(cfg.paths.dataset) / "templates" / "hints")
    out_dir = Path(cfg.paths.debug)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.images:
        img = cv2.imread(path)
        if img is None:
            print(f"{path}: cannot read")
            continue
        print(f"== {Path(path).name}")
        try:
            gv = read_gameplay(img, pal, cfg.vision, reader, log=lambda s: print("  " + s),
                               icons=icons, hints=hints)
        except VisionError as e:
            print(f"  not a gameplay frame? {e}")
            continue
        for line in gv.labels():
            print("  " + line)
        print(f"  tappable: {gv.tappable()}   conveyor: {gv.counter}/5")
        print("  slots   : " + ", ".join(f"{s.color}{'' if s.fill is None else f' {s.fill}%'}"
                                         f"{' (?)' if s.uncertain else ''}" for s in gv.slots))
        print("  sand    : " + ", ".join(f"{k} {v:.0%}" for k, v in
                                         sorted(gv.sand.items(), key=lambda kv: -kv[1])))
        print("  exposed : " + ", ".join(f"{k} {v:.0%}" for k, v in
                                         sorted(gv.exposed.items(), key=lambda kv: -kv[1])))
        print("  lower   : " + ", ".join(f"{k} {v:.0%}" for k, v in
                                         sorted(gv.lower.items(), key=lambda kv: -kv[1])))
        for note in gv.colour_notes:
            print("  colour  : " + note)
        p = out_dir / f"vision-{Path(path).stem}.png"
        cv2.imwrite(str(p), draw(img, gv))
        print(f"  saved   : {p}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
