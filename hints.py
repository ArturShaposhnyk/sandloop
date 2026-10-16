"""Game hint overlays the bot must ignore: the big yellow DOWN ARROWS the game draws over the
bottom of the grid, pointing at the boosters, while 4 buckets on the belt are not full yet
(L183, shot 055).

The arrows are not panel-coloured, so without masking they join the grid region to the booster
buttons (the lattice comes out too big and every cell is misplaced) and they read as orange
buckets. ``HintArrows.find`` matches a masked colour template (cut from shot 055, alpha =
arrow + outline) in the grid zone; ``occlusion_mask`` is the arrow pixels in the frame.
vision.read_gameplay keeps them out of the lattice, inpaints them before reading cells and
curtains, and marks cells mostly under an arrow as occluded (see ``CellMemory``).

    python -m hints cut <frame> x0 y0 x1 y1 [name]   # new template -> dataset/templates/hints/
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from cells import Cell
    from config import VisionConfig

Box = Tuple[int, int, int, int]
TEMPLATE_WIDTH = 602    # frame width the templates were cut at (pixels)


@dataclass
class HintArrows:
    templates: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)  # (bgr, mask)

    @classmethod
    def load(cls, directory: Path) -> "HintArrows":
        import cv2

        out = cls()
        for p in sorted(Path(directory).glob("*.png")):
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is not None and img.ndim == 3 and img.shape[2] == 4:
                out.templates.append((img[:, :, :3].copy(), img[:, :, 3] > 127))
        return out

    def find(self, img: np.ndarray, v: "VisionConfig") -> List[Tuple[Box, np.ndarray]]:
        """Arrows in the grid zone: (box in frame pixels, arrow mask of the box)."""
        import cv2

        if not self.templates:
            return []
        h, w = img.shape[:2]
        zx0, zy0 = int(v.grid_zone[0] * w), int(v.grid_zone[1] * h)
        zone = img[zy0:int(v.grid_zone[3] * h), zx0:int(v.grid_zone[2] * w)]
        k = v.hint_downscale     # match on a downscaled zone (speed), boxes in full pixels
        small = cv2.resize(zone, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        found: List[Tuple[float, Box, np.ndarray]] = []
        for bgr, mask in self.templates:
            for s in v.hint_scales:
                f = s * w / TEMPLATE_WIDTH
                full = cv2.resize(mask.astype(np.uint8), None, fx=f, fy=f,
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
                t = cv2.resize(bgr, None, fx=f * k, fy=f * k, interpolation=cv2.INTER_AREA)
                m = cv2.resize(mask.astype(np.uint8), (t.shape[1], t.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
                th, tw = t.shape[:2]
                if th > small.shape[0] or tw > small.shape[1]:
                    continue
                res = cv2.matchTemplate(small, t, cv2.TM_CCOEFF_NORMED, mask=m)
                res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
                fh, fw = full.shape
                while True:
                    _, score, _, (x, y) = cv2.minMaxLoc(res)
                    if score < v.hint_threshold:
                        break
                    bx, by = zx0 + int(round(x / k)), zy0 + int(round(y / k))
                    found.append((score, (bx, by, bx + fw, by + fh), full))
                    res[max(0, y - th // 2):y + th // 2 + 1,
                        max(0, x - tw // 2):x + tw // 2 + 1] = -1.0
        out: List[Tuple[Box, np.ndarray]] = []
        for score, box, m in sorted(found, key=lambda f: -f[0]):   # one per place, best first
            if not any(_overlap(box, b) > 0.3 for b, _ in out):
                out.append((box, m))
        return out


def _overlap(a: Box, b: Box) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / max(1, small)


def occlusion_mask(shape: Tuple[int, ...], arrows: List[Tuple[Box, np.ndarray]],
                   grow: int = 2) -> np.ndarray:
    """Frame-size bool mask of the arrow pixels (grown by ``grow`` px for anti-aliasing)."""
    import cv2

    out = np.zeros(shape[:2], np.uint8)
    for (x0, y0, x1, y1), m in arrows:
        out[y0:y1, x0:x1][m[:y1 - y0, :x1 - x0]] = 1
    if grow:
        out = cv2.dilate(out, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
    return out.astype(bool)


def clean(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The frame with the masked pixels inpainted from their surroundings."""
    import cv2

    if not mask.any():
        return img
    return cv2.inpaint(img, mask.astype(np.uint8), 5, cv2.INPAINT_TELEA)


class CellMemory:
    """The last clean reading of every grid cell, to stand in for cells hidden by a hint arrow.
    Valid only while the lattice is the same (same rows / cols / position within half a cell)."""

    def __init__(self) -> None:
        self._key: Optional[Tuple[int, int, int, int]] = None
        self._cells: Dict[Tuple[int, int], "Cell"] = {}

    @staticmethod
    def key(lat) -> Tuple[int, int, int, int]:
        return (lat.rows, lat.cols, int(round(lat.x0 / (lat.cw / 2))),
                int(round(lat.y0 / (lat.ch / 2))))

    def recall(self, lat, pos: Tuple[int, int]) -> Optional["Cell"]:
        if self.key(lat) != self._key:
            return None
        cell = self._cells.get(pos)
        return None if cell is None else copy.deepcopy(cell)

    def store(self, lat, grid) -> None:
        """Remember the clean (not occluded) cells of a successful read."""
        k = self.key(lat)
        if k != self._key:
            self._key, self._cells = k, {}
        for row in grid:
            for cell in row:
                if cell is not None and not cell.occluded:
                    self._cells[(cell.row, cell.col)] = copy.deepcopy(cell)


def cut_template(img: np.ndarray, box: Box, tol: float = 25.0, outline: int = 3
                 ) -> np.ndarray:
    """BGRA template of the arrow in ``box``: alpha = pixels near the fill colour sampled at the
    box centre (largest blob, holes filled) grown by ``outline`` px for the dark outline."""
    import cv2

    from palette import bgr_to_lab, lab_distance

    x0, y0, x1, y1 = box
    crop = img[y0:y1, x0:x1]
    lab = bgr_to_lab(crop)
    h, w = crop.shape[:2]
    fill = tuple(np.median(lab[h // 2 - 3:h // 2 + 3, w // 2 - 3:w // 2 + 3].reshape(-1, 3),
                           axis=0))
    m = (lab_distance(lab, fill) < tol).astype(np.uint8)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        raise ValueError("no arrow blob in the box")
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    blob = (lbl == k).astype(np.uint8)
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(blob, cnts, -1, 1, cv2.FILLED)
    blob = cv2.dilate(blob, np.ones((2 * outline + 1, 2 * outline + 1), np.uint8))
    return np.dstack([crop, blob * 255]).astype(np.uint8)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    import cv2

    from config import load_config

    ap = argparse.ArgumentParser(prog="python -m hints")
    sub = ap.add_subparsers(dest="cmd", required=True)
    cut = sub.add_parser("cut", help="cut a hint-arrow template from a frame")
    cut.add_argument("frame")
    cut.add_argument("box", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"))
    cut.add_argument("name", nargs="?", default="arrow_down")
    find = sub.add_parser("find", help="list hint arrows found in frames")
    find.add_argument("frames", nargs="+")
    args = ap.parse_args(argv)
    cfg = load_config()
    out_dir = Path(cfg.paths.dataset) / "templates" / "hints"
    if args.cmd == "cut":
        img = cv2.imread(args.frame)
        if img is None or img.shape[1] != TEMPLATE_WIDTH:
            print(f"{args.frame}: cannot read, or not {TEMPLATE_WIDTH} px wide")
            return 1
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{args.name}.png"
        cv2.imwrite(str(path), cut_template(img, tuple(args.box)))
        print(f"saved {path}")
        return 0
    arrows = HintArrows.load(out_dir)
    for p in args.frames:
        img = cv2.imread(p)
        boxes = [] if img is None else [b for b, _ in arrows.find(img, cfg.vision)]
        print(f"{Path(p).name}: {len(boxes)} arrow(s) {boxes}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
