"""Read small numbers on screen (k/5 counter, fill %, ice / curtain counters) by template
matching of single glyphs. Templates are cut from annotated dataset screenshots by
``tools/build_palette.py`` and stored in dataset/digits.npz.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

GLYPH_W, GLYPH_H = 12, 18


@dataclass
class Glyph:
    x0: int
    x1: int
    y0: int
    y1: int
    img: np.ndarray          # normalized GLYPH_H x GLYPH_W float32 0..1

    @property
    def aspect(self) -> float:
        return (self.x1 - self.x0) / max(1, self.y1 - self.y0)


def glyph_mask(roi_bgr: np.ndarray, polarity: str = "light") -> np.ndarray:
    """Pixels of the text: 'light' = white text (with a dark outline), 'dark' = dark text on a
    light sign."""
    import cv2

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    if polarity == "light":
        return (hsv[..., 1] < 70) & (hsv[..., 2] > 200)
    return hsv[..., 2] < 110


def segment(roi_bgr: np.ndarray, polarity: str = "light", min_rel_h: float = 0.55) -> List[Glyph]:
    """Glyphs left to right. Components touching the ROI border (frames, rails) are dropped;
    components mostly overlapping in x are merged (the parts of '%'); then anything much
    shorter than the tallest glyph (sand grains, specks) is dropped."""
    import cv2

    mask = glyph_mask(roi_bgr, polarity).astype(np.uint8)
    H, W = mask.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area < 4 or x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        boxes.append([x, x + w, y, y + h])
    boxes.sort()
    merged: List[List[int]] = []
    for b in boxes:
        overlap = min(b[1], merged[-1][1]) - b[0] if merged else 0
        if merged and overlap > 0.5 * min(b[1] - b[0], merged[-1][1] - merged[-1][0]):
            m = merged[-1]
            m[1], m[2], m[3] = max(m[1], b[1]), min(m[2], b[2]), max(m[3], b[3])
        else:
            merged.append(list(b))
    if not merged:
        return []
    max_h = max(b[3] - b[2] for b in merged)
    out = []
    for x0, x1, y0, y1 in merged:
        if y1 - y0 < min_rel_h * max_h or (x1 - x0) > 2.2 * (y1 - y0):
            continue
        crop = mask[y0:y1, x0:x1].astype(np.float32)
        img = cv2.resize(crop, (GLYPH_W, GLYPH_H), interpolation=cv2.INTER_AREA)
        out.append(Glyph(int(x0), int(x1), int(y0), int(y1), img))
    return out


def group_words(glyphs: List[Glyph], gap_rel: float = 0.6) -> List[List[Glyph]]:
    """Split glyphs into words where the horizontal gap exceeds ``gap_rel`` x glyph height."""
    words: List[List[Glyph]] = []
    for g in glyphs:
        if words and g.x0 - words[-1][-1].x1 <= gap_rel * (g.y1 - g.y0):
            words[-1].append(g)
        else:
            words.append([g])
    return words


@dataclass
class DigitReader:
    templates: Dict[str, List[np.ndarray]] = field(default_factory=dict)  # char -> images
    aspects: Dict[str, List[float]] = field(default_factory=dict)

    def add(self, ch: str, g: Glyph) -> None:
        self.templates.setdefault(ch, []).append(g.img)
        self.aspects.setdefault(ch, []).append(g.aspect)

    def classify(self, g: Glyph) -> Tuple[str, float]:
        best, best_d = "?", float("inf")
        for ch, imgs in self.templates.items():
            for img, asp in zip(imgs, self.aspects[ch]):
                d = float(np.abs(img - g.img).mean()) + 0.3 * abs(asp - g.aspect)
                if d < best_d:
                    best, best_d = ch, d
        return best, best_d

    def read_glyphs(self, glyphs: List[Glyph], max_dist: float = 0.35) -> str:
        out = []
        for g in glyphs:
            ch, d = self.classify(g)
            out.append(ch if d <= max_dist else "?")
        return "".join(out)

    def read(self, roi_bgr: np.ndarray, polarity: str = "light") -> str:
        return self.read_glyphs(segment(roi_bgr, polarity))

    def read_int(self, roi_bgr: np.ndarray, polarity: str = "light") -> Optional[int]:
        s = self.read(roi_bgr, polarity)
        return int(s) if s.isdigit() else None

    def save(self, path: Union[str, Path]) -> None:
        arrays = {}
        for ch, imgs in self.templates.items():
            key = f"c{ord(ch)}"
            arrays[key] = np.stack(imgs)
            arrays[key + "_aspect"] = np.asarray(self.aspects[ch], dtype=np.float32)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DigitReader":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"digit templates not found: {p} — build them with "
                                    "`python -m tools.build_palette`")
        data = np.load(p)
        r = cls()
        for key in data.files:
            if key.endswith("_aspect"):
                continue
            ch = chr(int(key[1:]))
            r.templates[ch] = list(data[key])
            r.aspects[ch] = [float(a) for a in data[key + "_aspect"]]
        return r
