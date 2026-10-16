"""Read the gameplay state from a frame of the iPhone screen (BGR, any resolution).

Everything is located relative to the screen (zones in config.yaml ``vision``) or detected
(grid lattice, curtains); colours come from the dataset palette (palette.py). Output positions
are relative 0..1 coordinates, ready for actions.Tapper.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cells import Cell, Kind, cell_features, classify_cell, empty_fraction, path_playable
from config import VisionConfig
from digits import DigitReader, group_words, segment
from palette import Palette, bgr_to_lab, lab_distance

Grid = List[List[Optional[Cell]]]   # None = no grid cell at this lattice position


def crop_rel(img: np.ndarray, r: Sequence[float]) -> np.ndarray:
    h, w = img.shape[:2]
    return img[int(r[1] * h):int(r[3] * h), int(r[0] * w):int(r[2] * w)]


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


def find_lattice(img: np.ndarray, palette: Palette, v: VisionConfig) -> Lattice:
    """Find the grid panel (the hole in the purple background below the conveyor) and its
    cell lattice. Column pitch comes from the periodic cell edges; row pitch from the fixed
    cell aspect ratio."""
    import cv2

    palette.require_ui("panel_bg")
    h, w = img.shape[:2]
    zx0, zy0 = int(v.grid_zone[0] * w), int(v.grid_zone[1] * h)
    zone = crop_rel(img, v.grid_zone)
    panel = lab_distance(bgr_to_lab(zone), palette.ui["panel_bg"]) < v.panel_tol
    side = panel[:, : max(2, int(0.05 * w))].mean(axis=1) > 0.9
    if not side.any():
        raise VisionError("grid panel background not found")
    ytop = int(np.argmax(side))
    grid = ~panel
    grid[:ytop] = False
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
    mask = np.zeros((rows, cols), bool)
    for r in range(rows):
        for c in range(cols):
            a = g[int(gy0 + (r + .25) * ch):int(gy0 + (r + .75) * ch),
                  int(gx0 + (c + .25) * cw):int(gx0 + (c + .75) * cw)]
            mask[r, c] = a.size > 0 and a.mean() > 0.5
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
    regions: List[Region] = []
    used = np.zeros(gold.shape[0], bool)
    for y in range(gold.shape[0]):
        if used[y]:
            continue
        row = np.concatenate([[0], gold[y].astype(np.int8), [0]])
        d = np.diff(row)
        starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
        runs = [(s, e) for s, e in zip(starts, ends) if e - s >= min_run]
        for s, e in runs:
            post = max(2, (e - s) // 25)
            sides = gold[y:, s:s + post].any(axis=1) | gold[y:, e - post:e].any(axis=1)
            bottom = y + int(np.nonzero(sides)[0].max()) + 1
            if bottom - y < 0.8 * lat.ch:
                continue
            used[y:bottom] = True
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


def cell_crop(img: np.ndarray, lat: Lattice, r: int, c: int, v: VisionConfig) -> np.ndarray:
    """The cell's rectangle shifted up by ``cell_shift_up`` (objects are drawn overhanging into
    the row above) and inset by ``cell_inset`` on each side."""
    x0, y0, x1, y1 = lat.rect(r, c)
    dy = v.cell_shift_up * lat.ch
    ix, iy = v.cell_inset * lat.cw, v.cell_inset * lat.ch
    return img[max(0, int(y0 - dy + iy)):int(y1 - dy - iy), int(x0 + ix):int(x1 - ix)]


def lock_digits_roi(img: np.ndarray, lat: Lattice, r: int, c: int, v: VisionConfig) -> np.ndarray:
    """Where a single-cell lock shows its counter: the cell shifted up like the objects."""
    x0, y0, x1, y1 = lat.rect(r, c)
    dy = v.cell_shift_up * lat.ch
    return img[max(0, int(y0 - dy)):int(y1 - dy / 2), int(x0):int(x1)]


@dataclass
class Slot:
    x: float                    # relative x centre on the screen
    color: Optional[str]
    fill: Optional[int] = None  # percent, if its label was read

    @property
    def uncertain(self) -> bool:
        """At the far edge without a fill label: entering/leaving bucket or an end post.
        The k/5 counter, not the slot list, is authoritative for how full the conveyor is."""
        return self.fill is None and not 0.12 <= self.x <= 0.88


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
    frame_size: Tuple[int, int] = (0, 0)       # w, h in pixels

    def labels(self) -> List[str]:
        return [" ".join("x" if c is None else c.label() for c in row) for row in self.grid]

    def tappable(self) -> List[Tuple[int, int]]:
        """Buckets that are playable by the path rule AND look face-up (both must agree)."""
        return [(r, c) for r, row in enumerate(self.grid) for c, cell in enumerate(row)
                if cell is not None and cell.kind is Kind.BUCKET and self.playable[r][c]
                and cell.face_up]

    def cell_center_rel(self, r: int, c: int) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.lattice.rect(r, c)
        w, h = self.frame_size
        return (x0 + x1) / 2 / w, (y0 + y1) / 2 / h

    def unknown_cells(self) -> List[Cell]:
        return [c for row in self.grid for c in row if c is not None and c.kind is Kind.UNKNOWN]


def read_grid(img: np.ndarray, palette: Palette, v: VisionConfig,
              reader: Optional[DigitReader] = None) -> Tuple[Lattice, Grid, List[Region]]:
    lat = find_lattice(img, palette, v)
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
            cell = classify_cell(r, c, f, palette, v)
            if cell.kind is Kind.LOCK and reader is not None:
                cell.counter = reader.read_int(lock_digits_roi(img, lat, r, c, v), "light")
            grid[r][c] = cell
    return lat, grid, regions


def read_conveyor(img: np.ndarray, palette: Palette, v: VisionConfig,
                  reader: Optional[DigitReader] = None) -> Tuple[Optional[int], List[Slot]]:
    """k of the "k/5" counter, and the buckets visible on the belt with their fill labels."""
    import cv2

    h, w = img.shape[:2]
    k = None
    if reader is not None:
        s = reader.read(crop_rel(img, v.counter))
        if len(s) == 3 and s[1] == "/" and s[0].isdigit():
            k = int(s[0])
    band = crop_rel(img, v.conveyor)
    lab = bgr_to_lab(band)
    # belt and rails are uniform along each row and buckets cover well under half of it, so the
    # row median is the background; buckets are what differs from it
    background = np.median(lab, axis=1, keepdims=True)
    obj = np.linalg.norm(lab - background, axis=2) > 18
    cols = obj.mean(axis=0) > 0.15
    padded = np.concatenate([[0], cols.astype(np.int8), [0]])
    d = np.diff(padded)
    runs: List[List[int]] = []
    for s, e in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
        if runs and s - runs[-1][1] < 0.03 * w:   # a bucket full of light sand looks hollow
            runs[-1][1] = e
        else:
            runs.append([s, e])
    bx0 = int(v.conveyor[0] * w)
    slots: List[Slot] = []
    for s, e in runs:
        m = obj[:, s:e]
        if e - s < 0.05 * w or m.mean() < v.slot_min_cover:   # too narrow, or an end post
            continue
        color = None
        if m.sum() >= 20:
            sel = lab[:, s:e][m]
            color, _ = palette.nearest(tuple(np.median(sel, axis=0)), "buckets",
                                       1.5 * v.color_max_dist)
        slots.append(Slot((bx0 + (s + e) / 2) / w, color))
    if reader is not None and slots:
        text = crop_rel(img, v.fill_text)
        for word in group_words(segment(text)):
            s = reader.read_glyphs(word)
            if not (s.endswith("%") and s[:-1].isdigit()):
                continue
            xc = (word[0].x0 + word[-1].x1) / 2 / w
            slot = min(slots, key=lambda sl: abs(sl.x - xc))
            if abs(slot.x - xc) < 0.06:
                slot.fill = int(s[:-1])
    return k, slots


def read_sand(img: np.ndarray, palette: Palette, v: VisionConfig
              ) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Share of the picture covered by each sand colour, and the same for the bottom band
    (sand a bucket can collect right now)."""
    pic = crop_rel(img, v.picture)
    lab = bgr_to_lab(pic)
    idx, names = palette.classify_pixels(lab, "sand", v.sand_max_dist)
    if "picture_empty" in palette.ui:
        idx[lab_distance(lab, palette.ui["picture_empty"]) < 10] = -1
    total = idx.size
    band = idx[int((1 - v.bottom_band) * idx.shape[0]):]
    sand = {n: float((idx == i).sum()) / total for i, n in enumerate(names)}
    exposed = {n: float((band == i).sum()) / max(1, band.size) for i, n in enumerate(names)}
    return ({n: f for n, f in sand.items() if f > 0.002},
            {n: f for n, f in exposed.items() if f > 0.01})


def read_gameplay(img: np.ndarray, palette: Palette, v: VisionConfig,
                  reader: Optional[DigitReader] = None,
                  log: Callable[[str], None] = print) -> GameView:
    lat, grid, regions = read_grid(img, palette, v, reader)
    playable = path_playable(grid, lat.exits())
    disagree = []
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell is not None and cell.kind is Kind.BUCKET and cell.face_up != playable[r][c]:
                disagree.append((r, c))
                log(f"warning: cell r{r}c{c} ({cell.label()}) looks "
                    f"{'face-up' if cell.face_up else 'face-down'} but the path rule says "
                    f"{'playable' if playable[r][c] else 'closed'} — not tapping it")
    k, slots = read_conveyor(img, palette, v, reader)
    sand, exposed = read_sand(img, palette, v) if palette.sand else ({}, {})
    return GameView(lat, grid, regions, playable, disagree, k, slots, sand, exposed,
                    (img.shape[1], img.shape[0]))


def save_unknown(img: np.ndarray, view: GameView, out_dir: Path,
                 log: Callable[[str], None] = print) -> List[Path]:
    """Save the frame + a crop of every UNKNOWN cell to out_dir and append to unknown.jsonl,
    so the user can explain the new obstacle and it can be added as a lock type."""
    import cv2

    cells = view.unknown_cells()
    if not cells:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    frame_path = out_dir / f"{stamp}-frame.png"
    cv2.imwrite(str(frame_path), img)
    saved = [frame_path]
    with (out_dir / "unknown.jsonl").open("a", encoding="utf-8") as fh:
        for cell in cells:
            x0, y0, x1, y1 = (int(round(a)) for a in view.lattice.rect(cell.row, cell.col))
            pad = int(0.4 * view.lattice.ch)
            crop = img[max(0, y0 - pad):y1, max(0, x0):x1]
            p = out_dir / f"{stamp}-r{cell.row}c{cell.col}.png"
            cv2.imwrite(str(p), crop)
            saved.append(p)
            rec = {"time": stamp, "frame": frame_path.name, "crop": p.name, "row": cell.row,
                   "col": cell.col, "reason": cell.reason,
                   "center_rel": [round(x, 4) for x in view.cell_center_rel(cell.row, cell.col)]}
            fh.write(json.dumps(rec) + "\n")
            log(f"unknown cell r{cell.row}c{cell.col} ({cell.reason}) saved to {p}")
    return saved


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
    out_dir = Path(cfg.paths.debug)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.images:
        img = cv2.imread(path)
        if img is None:
            print(f"{path}: cannot read")
            continue
        print(f"== {Path(path).name}")
        try:
            gv = read_gameplay(img, pal, cfg.vision, reader, log=lambda s: print("  " + s))
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
        p = out_dir / f"vision-{Path(path).stem}.png"
        cv2.imwrite(str(p), draw(img, gv))
        print(f"  saved   : {p}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
