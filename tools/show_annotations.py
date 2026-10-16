"""Draw the hand-annotated grids from dataset/annotations.json on their screenshots, so the
annotations can be checked by eye.

    python -m tools.show_annotations          # writes debug/annotations/<seq>.png

Each cell shows its annotated label; the box colour is the annotated bucket colour (taken from
the palette), cyan for ice, magenta for curtains, grey for empty cells. Face-up buckets get a
thick box, face-down a thin one; cells where the annotation disagrees with the path rule are
marked with a yellow "!".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from config import load_config
from palette import Palette
from tools.build_palette import grid_labels, load_annotations, parse_label, read_shot
from vision import find_lattice

LOCK_BGR = {"ice": (230, 200, 60), "curtain": (200, 60, 200), "empty": (170, 170, 170)}


def lab_to_bgr(lab: Sequence[float]) -> Tuple[int, int, int]:
    import cv2

    px = np.array([[[lab[0] * 255 / 100, lab[1] + 128, lab[2] + 128]]], dtype=np.uint8)
    b, g, r = cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0, 0]
    return int(b), int(g), int(r)


def draw_grid(img: np.ndarray, rows, lat, pal: Palette, disagree) -> np.ndarray:
    import cv2

    out = img.copy()
    for r, row in enumerate(rows):
        for c, label in enumerate(row):
            kind, color, body, up, _ = parse_label(label)
            if kind == "none":
                continue
            x0, y0, x1, y1 = (int(round(a)) for a in lat.rect(r, c))
            if kind == "bucket":
                bgr = lab_to_bgr(np.mean(pal.buckets[color], axis=0)) if color in pal.buckets \
                    else (0, 0, 255)
                width = 4 if up else 1
            else:
                bgr, width = LOCK_BGR.get(kind, (0, 0, 255)), 2
            cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 0), width + 2)
            cv2.rectangle(out, (x0, y0), (x1, y1), bgr, width)
            for i, part in enumerate(label.replace("-", " ").split()):
                org = (x0 + 3, y0 + 13 + 12 * i)
                cv2.putText(out, part, org, cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 3,
                            cv2.LINE_AA)
                cv2.putText(out, part, org, cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1,
                            cv2.LINE_AA)
            if (r, c) in disagree:
                cv2.putText(out, "!", (x1 - 14, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 230, 255), 2, cv2.LINE_AA)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    import cv2

    ap = argparse.ArgumentParser(prog="python -m tools.show_annotations", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    dataset = Path(cfg.paths.dataset)
    ann = load_annotations(dataset)
    pal = Palette.load(cfg.paths.palette)
    out_dir = Path(cfg.paths.debug) / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)
    for seq, rows in grid_labels(ann).items():
        img = read_shot(dataset, seq)
        lat = find_lattice(img, pal, cfg.vision)
        disagree = {tuple(x) for x in ann.get("visual_vs_path_disagreements", {}).get(seq, [])}
        out = draw_grid(img, rows, lat, pal, disagree)
        h = img.shape[0]
        top = max(0, int(lat.y0 - 0.12 * h))   # include the conveyor and k/5 above the grid
        bottom = min(h, int(lat.y0 + lat.rows * lat.ch + 0.02 * h))
        p = out_dir / f"{seq}.png"
        cv2.imwrite(str(p), out[top:bottom])
        print(f"saved {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
