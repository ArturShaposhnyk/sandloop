"""Colour palette built from the dataset (dataset/palette.json), and LAB colour matching.

No RGB values are hardcoded in the vision code: every colour it compares against (bucket colours,
sand colours, UI colours such as the panel background) comes from this palette, which
``tools/build_palette.py`` builds from hand annotations of dataset screenshots.

LAB here is CIELAB with L in 0..100 and a, b in about -128..127 (OpenCV's 8-bit LAB rescaled).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

Lab = Tuple[float, float, float]


def bgr_to_lab(pixels: np.ndarray) -> np.ndarray:
    """BGR uint8 array (..., 3) -> float32 LAB array of the same shape."""
    import cv2

    arr = np.ascontiguousarray(pixels, dtype=np.uint8)
    flat = arr.reshape(-1, 1, 3)
    lab = cv2.cvtColor(flat, cv2.COLOR_BGR2LAB).reshape(arr.shape).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab


def lab_distance(lab: np.ndarray, ref: Sequence[float]) -> np.ndarray:
    """Euclidean LAB distance of every pixel in ``lab`` (..., 3) to one reference colour."""
    return np.linalg.norm(lab - np.asarray(ref, dtype=np.float32), axis=-1)


def median_lab(pixels_bgr: np.ndarray) -> Lab:
    lab = bgr_to_lab(pixels_bgr).reshape(-1, 3)
    return tuple(float(v) for v in np.median(lab, axis=0))  # type: ignore[return-value]


@dataclass
class Palette:
    """``buckets`` / ``sand``: colour name -> list of LAB centres (several looks per colour,
    e.g. a face-up and a face-down bucket). ``ui``: UI element name -> one LAB centre."""

    buckets: Dict[str, List[Lab]] = field(default_factory=dict)
    sand: Dict[str, List[Lab]] = field(default_factory=dict)
    ui: Dict[str, Lab] = field(default_factory=dict)

    def nearest(self, lab: Sequence[float], group: str = "buckets",
                max_dist: Optional[float] = None) -> Tuple[Optional[str], float]:
        """Closest colour name in ``group`` and its distance; name is None if beyond max_dist."""
        best, best_d = None, float("inf")
        p = np.asarray(lab, dtype=np.float32)
        for name, centres in getattr(self, group).items():
            for c in centres:
                d = float(np.linalg.norm(p - np.asarray(c, dtype=np.float32)))
                if d < best_d:
                    best, best_d = name, d
        if max_dist is not None and best_d > max_dist:
            return None, best_d
        return best, best_d

    def classify_pixels(self, lab: np.ndarray, group: str = "sand",
                        max_dist: float = 18.0) -> Tuple[np.ndarray, List[str]]:
        """Per-pixel nearest colour index into the returned name list (-1 = no colour within
        max_dist). ``lab`` is (..., 3)."""
        names = list(getattr(self, group))
        out = np.full(lab.shape[:-1], -1, dtype=np.int16)
        best = np.full(lab.shape[:-1], np.inf, dtype=np.float32)
        for i, name in enumerate(names):
            for c in getattr(self, group)[name]:
                d = lab_distance(lab, c)
                better = (d < best) & (d <= max_dist)
                out[better], best[better] = i, d[better]
                best = np.minimum(best, d)
        return out, names

    def require_ui(self, *names: str) -> None:
        missing = [n for n in names if n not in self.ui]
        if missing:
            raise KeyError(f"palette has no UI colour(s) {missing} — rebuild it with "
                           "`python -m tools.build_palette`")

    def to_dict(self) -> dict:
        rnd = lambda c: [round(float(v), 2) for v in c]  # noqa: E731
        return {"buckets": {k: [rnd(c) for c in v] for k, v in sorted(self.buckets.items())},
                "sand": {k: [rnd(c) for c in v] for k, v in sorted(self.sand.items())},
                "ui": {k: rnd(v) for k, v in sorted(self.ui.items())}}

    @classmethod
    def from_dict(cls, d: dict) -> "Palette":
        tup = lambda c: tuple(float(v) for v in c)  # noqa: E731
        return cls(buckets={k: [tup(c) for c in v] for k, v in d.get("buckets", {}).items()},
                   sand={k: [tup(c) for c in v] for k, v in d.get("sand", {}).items()},
                   ui={k: tup(v) for k, v in d.get("ui", {}).items()})

    def save(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=1) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Palette":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"palette not found: {p} — build it with "
                                    "`python -m tools.build_palette`")
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
