"""Crops of the screen by relative zones (config.yaml ``vision``)."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def crop_rel(img: np.ndarray, r: Sequence[float]) -> np.ndarray:
    """The part of ``img`` inside the relative rectangle ``r`` = [x0, y0, x1, y1] (0..1)."""
    h, w = img.shape[:2]
    return img[int(r[1] * h):int(r[3] * h), int(r[0] * w):int(r[2] * w)]
