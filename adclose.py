"""Generic ad close button (✕) finder: an X-shaped glyph on a plain backdrop.

Ads differ in everything but the ✕, and it shows up late (GAME_RULES: wait for it). No template
fits them all, so the test is a shape: in each ``flow.close_zones`` corner zone, light-on-dark
and dark-on-light blobs (adaptive threshold) of ``close_min_size``..``close_max_size`` x the
screen width, roughly square, whose pixels overlap a drawn X by ``close_min_iou``, with all four
arms (corners), a filled centre and empty edge midpoints. Dataset: the App Store sheet ✕ (018)
0.75, popup ✕s 0.76-0.86; the mute icon's small "x" is below the size limit; no hit on
gameplay / menu / win screens. Only flow.py calls it, and only on ad screens.

``find_skip``: the button on a filled disc, top right, of some ads: ▶| skip (run 171133) or a
✕ too small for ``find_close`` (Kingshot end card, run 224631: 0.017 x the screen width).
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
    kind: str = "close"                        # "close" (✕) | "skip" (▶|)


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


def skip_score(glyph: np.ndarray) -> float:
    """``glyph``: binary light glyph cut to its box. A skip glyph (▶|, L187 run 171133 Trade
    Republic ad) has a solid vertical bar in its right third and a triangle that narrows to the
    right left of it; the mute speaker (top-left twin) has broken arcs there. Returns the bar's
    height share (0 when the layout does not fit)."""
    h, w = glyph.shape
    if h < 5 or w < 5:
        return 0.0
    right = glyph[:, int(w * 2 / 3):]
    bar = max((right[:, j].sum() for j in range(right.shape[1])), default=0) / h
    left = glyph[:, :int(w / 2)]
    tri = left.sum(axis=0)
    if bar < 0.85 or len(tri) < 2 or tri[0] < tri[-1]:   # the triangle's base is on the left
        return 0.0
    return float(bar)


def find_skip(img: np.ndarray, fc: FlowConfig) -> List[CloseMark]:
    """Buttons on a filled disc in ``flow.skip_zone`` - ▶| skip or ✕ (``close_min_iou``), best
    first."""
    import cv2

    H, W = img.shape[:2]
    z = fc.skip_zone
    x0, y0, x1, y1 = int(z[0] * W), int(z[1] * H), int(z[2] * W), int(z[3] * H)
    hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    if hsv.size == 0:
        return []
    disc = ((hsv[..., 1] >= 120) & (hsv[..., 2] >= 150)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(disc, connectivity=8)
    lo, hi = fc.skip_min_size * W, fc.skip_max_size * W
    out: List[CloseMark] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (lo <= w <= hi and lo <= h <= hi and 0.8 <= w / h <= 1.25):
            continue
        inner = hsv[y + h // 6:y + h - h // 6, x + w // 6:x + w - w // 6]
        light = ((inner[..., 1] < 90) & (inner[..., 2] >= 200)).astype(np.uint8)
        ys, xs = np.nonzero(light)
        if len(ys) < 6:
            continue
        glyph = light[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        score, kind = skip_score(glyph), "skip"
        if score == 0:
            score, kind = x_score(glyph), "close"
            if score < fc.close_min_iou:
                continue
        out.append(CloseMark(round(score, 3), (float((x0 + x + w / 2) / W),
                                               float((y0 + y + h / 2) / H)), float(w / W), kind))
    return sorted(out, key=lambda m: -m.score)
