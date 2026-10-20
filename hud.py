"""HUD reading outside the grid: lives (the white digit in the red heart, top bar).

The heart sits at the same place on the home map and in levels (also behind the dimmed fail /
revive popups). It counts only when palette ui ``heart_red`` covers ``flow.heart_min_share`` of
``flow.lives_zone`` (popups and ads cover it: None). The digit is read with the k/5 counter's
DigitReader (same game font; dataset: every heart reads 4 or 5 right).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from config import FlowConfig


def read_lives(img: np.ndarray, palette: Any, reader: Any, fc: FlowConfig) -> Optional[int]:
    from palette import bgr_to_lab, lab_distance
    from zones import crop_rel

    if "heart_red" not in palette.ui:
        return None
    roi = crop_rel(img, fc.lives_zone)
    if roi.size == 0:
        return None
    heart = lab_distance(bgr_to_lab(roi), palette.ui["heart_red"]) < fc.heart_tol
    if float(heart.mean()) < fc.heart_min_share:
        return None
    return reader.read_int(roi, "light")
