"""The conveyor: the "k/5" counter and the buckets riding the belt with their fill labels
(split out of vision.py)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from config import VisionConfig
from digits import DigitReader, group_words, segment
from palette import Palette, bgr_to_lab
from zones import crop_rel


@dataclass
class Slot:
    x: float                    # relative x centre on the screen
    color: Optional[str]        # the collecting bucket (a tower's current top stage)
    fill: Optional[int] = None  # percent, if its label was read
    body: Optional[str] = None  # tower: colour of the next stage (seen as its base), else None

    def label(self) -> str:
        return f"{self.color}/{self.body}" if self.body else f"{self.color}"

    @property
    def uncertain(self) -> bool:
        """At the far edge without a fill label: entering/leaving bucket or an end post.
        The k/5 counter, not the slot list, is authoritative for how full the conveyor is."""
        return self.fill is None and not 0.12 <= self.x <= 0.88


def read_counter(img: np.ndarray, v: VisionConfig, reader: DigitReader) -> Optional[int]:
    """k of the "k/5" counter. The zone is tight (light pixels right below the text merge with
    the glyphs in a taller one), so a crop a few px off used to lose it: the zone is slid up and
    down by up to ``counter_search`` and the reading nearest the configured position wins."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = v.counter
    step = max(1, int(0.003 * h))
    reach = int(v.counter_search * h)
    for dy in sorted(range(-reach, reach + 1, step), key=abs):
        top, bottom = int(y0 * h) + dy, int(y1 * h) + dy
        if top < 0 or bottom > h:
            continue
        s = reader.read(img[top:bottom, int(x0 * w):int(x1 * w)])
        if len(s) == 3 and s[1] == "/" and s[0].isdigit() and s[2].isdigit():
            return int(s[0])
    return None


def read_conveyor(img: np.ndarray, palette: Palette, v: VisionConfig,
                  reader: Optional[DigitReader] = None) -> Tuple[Optional[int], List[Slot]]:
    """k of the "k/5" counter, and the buckets visible on the belt with their fill labels."""
    h, w = img.shape[:2]
    k = None
    if reader is not None:
        k = read_counter(img, v, reader)
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
        slots.append(Slot((bx0 + (s + e) / 2) / w, color,
                          body=_belt_base(band[:, s:e], lab[:, s:e], m, color, palette, v)))
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


def _belt_base(bgr: np.ndarray, lab: np.ndarray, m: np.ndarray, color: Optional[str],
               palette: Palette, v: VisionConfig) -> Optional[str]:
    """A tower on the belt: the next stage shows as a base of another colour under the
    collecting bucket (L181: cream on red, red on orange). Base = the brighter half of the
    bucket pixels in the lowest 30 % of the belt zone (middle of the run). None for a one-colour bucket. A bucket
    still dropping onto the belt has its base below the zone (shot 031): read as one colour."""
    if color is None:
        return None
    base = m.copy()
    base[: int(0.7 * m.shape[0])] = False
    cut = int(0.2 * m.shape[1])           # middle 60 % of the run: no rails / neighbours
    base[:, :cut] = base[:, m.shape[1] - cut:] = False
    if base.sum() < 20:
        return None
    vals = bgr.max(axis=2)[base]
    sel = lab[base][vals >= np.percentile(vals, 50)]
    name, _ = palette.nearest(tuple(np.median(sel, axis=0)), "buckets", v.color_max_dist)
    return name if name is not None and name != color else None
