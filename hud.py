"""HUD reading outside the grid: lives (the white digit in the red heart, top bar) and the
belt speed button (▶▶2x under the k/5 counter: green = 2x on, grey = off).

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


def read_speed(img: np.ndarray, bc: Any) -> Optional[bool]:
    """The ▶▶2x button (``bot.speed_zone``): True = 2x on (green), False = off (grey), None =
    not readable (covered, animation, not gameplay). Live gameplay frames: on 0.34-0.57 green,
    off 0 green / ~0.4 grey."""
    import cv2

    from zones import crop_rel

    roi = crop_rel(img, bc.speed_zone)
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = float(((h >= bc.speed_hue[0]) & (h <= bc.speed_hue[1]) & (s >= bc.speed_min_sv)
                   & (v >= bc.speed_min_sv)).mean())
    if green >= bc.speed_on_share:
        return True
    grey = float(((s < 70) & (v >= 60) & (v <= 200)).mean())
    if green < 0.05 and grey >= bc.speed_off_grey:
        return False
    return None


class SpeedTracker:
    """The level's belt speed: a reading counts after ``frames`` equal ones in a row."""

    def __init__(self, frames: int) -> None:
        self.frames = frames
        self.state: Optional[bool] = None       # confirmed: True 2x, False 1x, None unknown
        self._last: Optional[bool] = None
        self._count = 0

    def feed(self, reading: Optional[bool]) -> bool:
        """Returns True when the confirmed state changed."""
        if reading is None:
            return False
        self._count = self._count + 1 if reading == self._last else 1
        self._last = reading
        if self._count >= self.frames and reading != self.state:
            self.state = reading
            return True
        return False
