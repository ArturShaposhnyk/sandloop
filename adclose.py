"""Generic ad close button (✕) finder: an X-shaped glyph on a plain backdrop.

Ads differ in everything but the ✕, and it shows up late (GAME_RULES: wait for it). No template
fits them all, so the test is a shape: in each ``flow.close_zones`` corner zone, light-on-dark
and dark-on-light blobs (adaptive threshold) of ``close_min_size``..``close_max_size`` x the
screen width, roughly square, whose pixels overlap a drawn X by ``close_min_iou``, with all four
arms (corners), a filled centre and empty edge midpoints. Dataset: the App Store sheet ✕ (018)
0.75, popup ✕s 0.76-0.86; the mute icon's small "x" is below the size limit; no hit on
gameplay / menu / win screens. Only flow.py calls it, and only on ad screens.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from config import FlowConfig


@dataclass
class CloseMark:
    score: float                               # overlap with a drawn X
    center_rel: Tuple[float, float]
    size_rel: float                            # glyph width, share of the screen width


def _x_mask(w: int, h: int, thickness: float) -> np.ndarray:
    import cv2

    m = np.zeros((h, w), np.uint8)
    t = max(1, int(round(thickness * min(w, h))))
    cv2.line(m, (0, 0), (w - 1, h - 1), 1, t)
    cv2.line(m, (w - 1, 0), (0, h - 1), 1, t)
    return m


def _share(comp: np.ndarray, y0: float, y1: float, x0: float, x1: float) -> float:
    h, w = comp.shape
    a, b = int(y0 * h), max(int(y0 * h) + 1, int(y1 * h))
    c, d = int(x0 * w), max(int(x0 * w) + 1, int(x1 * w))
    return float(comp[a:b, c:d].mean())


def x_score(comp: np.ndarray) -> float:
    """0 unless ``comp`` (binary glyph cut to its box) has the X layout; else its best overlap
    with a drawn X (three stroke widths)."""
    corners = min(_share(comp, 0, .25, 0, .25), _share(comp, 0, .25, .75, 1),
                  _share(comp, .75, 1, 0, .25), _share(comp, .75, 1, .75, 1))
    centre = _share(comp, .4, .6, .4, .6)
    mids = max(_share(comp, 0, .2, .4, .6), _share(comp, .8, 1, .4, .6),
               _share(comp, .4, .6, 0, .2), _share(comp, .4, .6, .8, 1))
    if corners <= 0.2 or centre <= 0.5 or mids >= 0.15:
        return 0.0
    h, w = comp.shape
    best = 0.0
    for t in (0.12, 0.18, 0.25):
        xm = _x_mask(w, h, t)
        best = max(best, float((comp & xm).sum()) / max(1, int((comp | xm).sum())))
    return best


def find_close(img: np.ndarray, fc: FlowConfig,
               zones: Sequence[Sequence[float]] = ()) -> List[CloseMark]:
    """✕ candidates in the iPhone screen image, best first."""
    import cv2

    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lo, hi = fc.close_min_size * W, fc.close_max_size * W
    out: List[CloseMark] = []
    for z in zones or fc.close_zones:
        x0, y0, x1, y1 = int(z[0] * W), int(z[1] * H), int(z[2] * W), int(z[3] * H)
        roi = gray[y0:y1, x0:x1]
        if roi.shape[0] < 31 or roi.shape[1] < 31:
            continue
        for r in (roi, 255 - roi):
            bw = cv2.adaptiveThreshold(r, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31,
                                       -fc.close_contrast)
            n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
            for i in range(1, n):
                x, y, w, h, _a = stats[i]
                if not (lo <= w <= hi and lo <= h <= hi and 0.75 <= w / h <= 1.33):
                    continue
                score = x_score((lab[y:y + h, x:x + w] == i).astype(np.uint8))
                if score >= fc.close_min_iou:
                    out.append(CloseMark(round(score, 3),
                                         ((x0 + x + w / 2) / W, (y0 + y + h / 2) / H), w / W))
    return sorted(out, key=lambda m: -m.score)
