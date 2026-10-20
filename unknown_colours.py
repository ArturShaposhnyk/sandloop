"""Colours missing from the palette: report them, never drop them silently.

A bucket / lock / key cell whose colour matches nothing is UNKNOWN (never tapped); sand colours
that match nothing are found as compact clusters of unmatched picture pixels. Both are saved to
dataset/unknown_colours/ so ``python -m tools.add_colour`` can add them to the palette.
UNKNOWN cells for other reasons (new obstacles) go to dataset/unknown/ (``save_unknown``).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import numpy as np

from config import VisionConfig
from palette import Palette, bgr_to_lab, lab_distance

if TYPE_CHECKING:
    from vision import GameView


@dataclass
class UnknownColour:
    """A colour that matches nothing in the palette: a bucket / lock / key cell, or a compact
    cluster of picture pixels (sand). Saved to dataset/unknown_colours/ for tools/add_colour."""
    kind: str                           # "bucket" | "lock" | "key" | "sand"
    lab: Tuple[float, float, float]
    share: float = 0.0                  # sand: share of the picture
    cell: Optional[Tuple[int, int]] = None

    def describe(self) -> str:
        lab = "/".join(f"{x:.0f}" for x in self.lab)
        where = (f" r{self.cell[0]}c{self.cell[1]}" if self.cell is not None
                 else f" {self.share:.0%} of the picture")
        return f"unknown {self.kind} colour LAB {lab}{where}"


def unknown_sand(img: np.ndarray, palette: Palette, v: VisionConfig,
                 clusters: int = 4, seed: int = 0,
                 mask: Optional[np.ndarray] = None) -> List[UnknownColour]:
    """Sand colours missing from the palette: picture pixels matching no sand colour (nor the
    empty picture, nor dark neutral shadow) are clustered; a cluster with at least
    ``unknown_sand_min`` of the picture within half of ``sand_max_dist`` of its centre is a
    colour. Scattered leftovers (grain edges, blends: <= 1.6 % per cluster) are not."""
    import cv2

    from vision import crop_rel

    lab = bgr_to_lab(crop_rel(img, v.picture))
    idx, _ = palette.classify_pixels(lab, "sand", v.sand_max_dist)
    un = idx < 0
    if "picture_empty" in palette.ui:
        un &= lab_distance(lab, palette.ui["picture_empty"]) >= 10
    un &= ~((lab[..., 0] < 35) & (np.hypot(lab[..., 1], lab[..., 2]) < 8))   # shadow
    if mask is not None:   # L188 lock boxes: their overlay is no sand colour (picture.lock_mask)
        un &= ~crop_rel(mask.astype(np.uint8)[..., None], v.picture)[..., 0].astype(bool)
    n = un.size
    px = lab[un]
    if len(px) < v.unknown_sand_min * n:
        return []
    rng = np.random.default_rng(seed)
    sub = px[rng.choice(len(px), min(len(px), 4000), replace=False)].astype(np.float32)
    k = min(clusters, len(sub))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    cv2.setRNGSeed(seed)
    _, _, centres = cv2.kmeans(sub, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    out: List[UnknownColour] = []
    for c in centres:
        share = float((np.linalg.norm(px - c, axis=1) < v.sand_max_dist / 2).sum()) / n
        if share >= v.unknown_sand_min and all(
                float(np.linalg.norm(c - np.asarray(o.lab))) >= v.sand_max_dist for o in out):
            out.append(UnknownColour("sand", tuple(float(x) for x in c), share))  # type: ignore[arg-type]
    return sorted(out, key=lambda u: -u.share)


def colour_key(u: UnknownColour, step: float = 6.0) -> Tuple[str, int, int, int]:
    """Same unknown colour across frames (for saving each one once per run)."""
    return (u.kind,) + tuple(int(round(x / step)) for x in u.lab)  # type: ignore[return-value]


def save_unknown_colours(img: np.ndarray, view: GameView, v: VisionConfig, out_dir: Path,
                         seen: Optional[set] = None,
                         log: Callable[[str], None] = print) -> List[Path]:
    """Save every unknown colour of the frame not in ``seen`` (updated) to out_dir: a cell crop
    (``<time>-<kind>-r<r>c<c>.png``) or, for sand, a patch of the matching picture pixels
    (``<time>-sand-<L>_<a>_<b>.png``), plus a line in unknown_colours.jsonl. Add a colour to
    the palette with ``python -m tools.add_colour <name> <files>``."""
    import cv2

    from vision import cell_crop, crop_rel

    seen = set() if seen is None else seen
    new = []
    for u in view.unknown_colours:   # one file per colour, also within a frame
        if colour_key(u) not in seen:
            seen.add(colour_key(u))
            new.append(u)
    if not new:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    saved = []
    with (out_dir / "unknown_colours.jsonl").open("a", encoding="utf-8") as fh:
        for u in new:
            if u.cell is not None:
                crop = cell_crop(img, view.lattice, u.cell[0], u.cell[1], v)
                p = out_dir / f"{stamp}-{u.kind}-r{u.cell[0]}c{u.cell[1]}.png"
            else:
                pic = crop_rel(img, v.picture)
                near = lab_distance(bgr_to_lab(pic), u.lab) < v.sand_max_dist / 2
                px = pic[near]
                side = max(1, int(np.sqrt(len(px))))
                crop = px[:side * side].reshape(side, side, 3)
                p = out_dir / f"{stamp}-sand-{'_'.join(f'{x:.0f}' for x in u.lab)}.png"
            if not cv2.imwrite(str(p), crop):
                log(f"unknown colour: cannot write {p}")
                continue
            saved.append(p)
            fh.write(json.dumps({"time": stamp, "file": p.name, "kind": u.kind,
                                 "lab": [round(x, 1) for x in u.lab],
                                 "share": round(u.share, 4),
                                 "cell": None if u.cell is None else list(u.cell)}) + "\n")
            log(f"{u.describe()} saved to {p}")
    return saved


def save_unknown(img: np.ndarray, view: GameView, out_dir: Path,
                 log: Callable[[str], None] = print) -> List[Path]:
    """Save the frame + a crop of every UNKNOWN cell to out_dir and append to unknown.jsonl,
    so the user can explain the new obstacle and it can be added as a lock type."""
    import cv2

    cells = [c for c in view.unknown_cells() if c.lab is None]   # colours: save_unknown_colours
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
