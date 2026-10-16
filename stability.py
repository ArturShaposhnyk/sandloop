"""Wait until the (animated) frame settles before making a decision. Pure numpy/cv2."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from config import StabilityConfig


def _prep(frame: np.ndarray, width: int) -> np.ndarray:
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    h, w = gray.shape[:2]
    if w > width:
        gray = cv2.resize(gray, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32)


def frame_diff(a: np.ndarray, b: np.ndarray, width: int = 160) -> float:
    """Mean absolute grayscale difference (0..255) between two frames, after downscaling."""
    pa, pb = _prep(a, width), _prep(b, width)
    if pa.shape != pb.shape:
        return float("inf")  # size changed (window resized) — definitely not stable
    return float(np.mean(np.abs(pa - pb)))


@dataclass
class StableResult:
    frame: np.ndarray
    stable: bool          # False = timed out, frame is just the latest one
    waited_s: float
    last_diff: float


def wait_stable(
    grab: Callable[[], np.ndarray],
    cfg: StabilityConfig,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> StableResult:
    """Grab frames until `cfg.consecutive` diffs in a row are below `cfg.threshold`."""
    start = clock()
    prev = grab()
    streak = 0
    diff: Optional[float] = None
    while True:
        if clock() - start >= cfg.timeout_s:
            return StableResult(prev, False, clock() - start, diff if diff is not None else float("inf"))
        sleep(cfg.poll_s)
        cur = grab()
        diff = frame_diff(prev, cur, cfg.downscale_width)
        streak = streak + 1 if diff < cfg.threshold else 0
        prev = cur
        if streak >= cfg.consecutive:
            return StableResult(cur, True, clock() - start, diff)
