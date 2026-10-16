"""Coordinate conversion — the single place where relative/pixel/point math lives.

Three coordinate spaces:
  * relative  (rx, ry) in 0..1 of the iPhone screen area — what all game logic uses;
  * pixel     (px, py) inside the captured *window* image (physical pixels, Retina-scaled);
  * point     (x, y)   global macOS display coordinates, top-left origin — what clicks use.

`ScreenGeometry` is built from the window bounds (points, from CGWindowList), the runtime
scale factor (captured pixels / window points) and the content rect (iPhone screen area as
fractions of the window). Only capture.py / actions.py should call the conversions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def as_int(self) -> Tuple[int, int, int, int]:
        return (round(self.x), round(self.y), round(self.w), round(self.h))


FULL_WINDOW = Rect(0.0, 0.0, 1.0, 1.0)


@dataclass(frozen=True)
class ScreenGeometry:
    window_pt: Rect             # window bounds, global points
    scale: float                # captured pixels per point (2.0 on Retina, detected at runtime)
    content_frac: Rect = FULL_WINDOW  # iPhone screen area as fractions of the window

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError("scale must be > 0")
        if self.window_pt.w <= 0 or self.window_pt.h <= 0:
            raise ValueError("window must have positive size")

    # --- derived rects -----------------------------------------------------------------
    @property
    def content_pt(self) -> Rect:
        """iPhone screen area in global points."""
        c, w = self.content_frac, self.window_pt
        return Rect(w.x + c.x * w.w, w.y + c.y * w.h, c.w * w.w, c.h * w.h)

    @property
    def content_px(self) -> Rect:
        """iPhone screen area in pixels of the captured window image."""
        c, w, s = self.content_frac, self.window_pt, self.scale
        return Rect(c.x * w.w * s, c.y * w.h * s, c.w * w.w * s, c.h * w.h * s)

    # --- relative <-> point ------------------------------------------------------------
    def rel_to_point(self, rx: float, ry: float) -> Tuple[float, float]:
        c = self.content_pt
        return (c.x + rx * c.w, c.y + ry * c.h)

    def point_to_rel(self, x: float, y: float) -> Tuple[float, float]:
        c = self.content_pt
        return ((x - c.x) / c.w, (y - c.y) / c.h)

    # --- relative <-> pixel (window image) ---------------------------------------------
    def rel_to_px(self, rx: float, ry: float) -> Tuple[float, float]:
        c = self.content_px
        return (c.x + rx * c.w, c.y + ry * c.h)

    def px_to_rel(self, px: float, py: float) -> Tuple[float, float]:
        c = self.content_px
        return ((px - c.x) / c.w, (py - c.y) / c.h)

    # --- relative <-> pixel (cropped content image, i.e. what vision sees) -------------
    def rel_to_content_px(self, rx: float, ry: float) -> Tuple[float, float]:
        c = self.content_px
        return (rx * c.w, ry * c.h)

    def content_px_to_rel(self, px: float, py: float) -> Tuple[float, float]:
        c = self.content_px
        return (px / c.w, py / c.h)

    # --- safety ------------------------------------------------------------------------
    def contains_point(self, x: float, y: float) -> bool:
        return self.content_pt.contains(x, y)


def is_valid_rel(rx: float, ry: float) -> bool:
    return 0.0 <= rx <= 1.0 and 0.0 <= ry <= 1.0


def detect_scale(image_width_px: int, window_width_pt: float) -> float:
    """Scale factor = captured pixels / window points. Snaps to 0.25 steps to absorb rounding."""
    if window_width_pt <= 0:
        raise ValueError("window width must be > 0")
    raw = image_width_px / window_width_pt
    snapped = round(raw * 4) / 4
    return snapped if snapped > 0 else raw


def in_corner(x: float, y: float, displays: list[Rect], margin: float) -> bool:
    """True if the global point (x, y) is within `margin` points of any display corner."""
    for d in displays:
        xs = (d.x, d.x + d.w - 1)
        ys = (d.y, d.y + d.h - 1)
        for cx in xs:
            for cy in ys:
                if abs(x - cx) <= margin and abs(y - cy) <= margin:
                    return True
    return False
