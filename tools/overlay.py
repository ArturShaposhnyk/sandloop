"""Live debug view of the iPhone screen with FPS and the relative coordinate under the cursor.

    python -m tools.overlay            # iPhone screen area only
    python -m tools.overlay --window   # whole window, detected screen rect drawn in green

Keys (overlay window focused): q / Esc quit, s save frame to debug/.
Global hotkeys from config.yaml (pause / quit) also work.
If the mirroring window is missing (or disappears), a placeholder is shown until it comes back.
With screen.capture=mss keep the overlay away from the mirroring window (it would capture itself).
"""
from __future__ import annotations

import argparse
import sys
import textwrap
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from capture import CaptureError, Grabber, WindowNotFound, WindowTracker, frame_change
from config import load_config

WIN_NAME = "Sand Loop overlay"
BAR_H = 30  # height of the bottom status bar (rel readout), in display pixels
QUIT_KEYS = (ord("q"), 27)


class Fps:
    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.value = 0.0
        self._t: Optional[float] = None

    def tick(self) -> float:
        now = time.perf_counter()
        if self._t is not None and now > self._t:
            inst = 1.0 / (now - self._t)
            self.value = inst if self.value == 0 else self.value + self.alpha * (inst - self.value)
        self._t = now
        return self.value


def _text(img: np.ndarray, s: str, org: Tuple[int, int]) -> None:
    import cv2

    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def clear_bar(img: np.ndarray, height: int = BAR_H) -> None:
    """Paint the bottom status bar solid black so the text drawn there never overlaps old text."""
    img[max(0, img.shape[0] - height):, :] = 0


def waiting_frame(msg: str, size: Tuple[int, int] = (480, 220)) -> np.ndarray:
    """Placeholder shown while the mirroring window is missing (keeps the GUI responsive)."""
    w, h = size
    img = np.full((h, w, 3), 32, dtype=np.uint8)
    _text(img, "waiting for iPhone Mirroring...", (10, 30))
    for i, line in enumerate(textwrap.wrap(msg, 50)[:6]):
        _text(img, line, (10, 65 + 25 * i))
    clear_bar(img)
    _text(img, "q/Esc quit", (10, h - 10))
    return img


def main(argv: Optional[Sequence[str]] = None) -> int:
    import cv2

    import permissions
    from safety import Hotkeys

    ap = argparse.ArgumentParser(prog="python -m tools.overlay", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", action="store_true", help="show the whole window + screen rect")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    permissions.require(screen=True, accessibility=False)

    grabber = Grabber(cfg.screen.capture)
    tracker = WindowTracker(cfg, grabber)
    hotkeys = Hotkeys(cfg.safety.hotkey_pause, cfg.safety.hotkey_quit).start()
    fps = Fps()
    mouse = {"pos": None}  # display-space cursor position
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN_NAME, lambda ev, x, y, *_: mouse.__setitem__("pos", (x, y)))
    prev: Optional[np.ndarray] = None
    frame: Optional[np.ndarray] = None
    print(f"overlay running — q/Esc to quit, s to save; hotkeys: pause={hotkeys.pause_spec} "
          f"quit={hotkeys.quit_spec}")
    waiting: Optional[str] = None
    try:
        while not hotkeys.quit.is_set():
            try:  # the window may be missing at start or vanish mid-run (closed, locked, minimized)
                geo = tracker.get(force=waiting is not None)
                if not hotkeys.paused.is_set() or frame is None:
                    if args.window:
                        frame = grabber.grab_window(tracker.window)[0].copy()
                        s = geo.scale
                        x0, y0 = (geo.content.x - geo.window.x) * s, (geo.content.y - geo.window.y) * s
                        origin = (x0, y0)
                        cv2.rectangle(frame, (int(x0), int(y0)),
                                      (int(x0 + geo.content.w * s), int(y0 + geo.content.h * s)),
                                      (0, 255, 0), 2)
                    else:
                        frame = tracker.grab()
                        origin = (0.0, 0.0)
            except (WindowNotFound, CaptureError) as e:
                if waiting is None:
                    print(f"waiting for window: {e}")
                waiting = str(e)
                prev = frame = None
                cv2.imshow(WIN_NAME, waiting_frame(waiting))
                if (cv2.waitKey(500) & 0xFF) in QUIT_KEYS:
                    break
                continue
            if waiting is not None:
                print("window found")
                waiting = None
            diff = frame_change(prev, frame) if prev is not None and prev.shape == frame.shape else 0.0
            prev = frame
            k = min(1.0, cfg.overlay.max_height / frame.shape[0])
            view = cv2.resize(frame, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else frame.copy()
            _text(view, f"FPS {fps.tick():5.1f}  diff {diff:.4f}  scale {geo.scale:.2f}", (10, 25))
            _text(view, f"screen {geo.content.w:.0f}x{geo.content.h:.0f}pt @ "
                        f"({geo.content.x:.0f},{geo.content.y:.0f})", (10, 50))
            if hotkeys.paused.is_set():
                _text(view, "PAUSED", (10, 75))
            clear_bar(view)
            if mouse["pos"] is not None:
                mx, my = mouse["pos"]
                rx, ry = geo.pixel_to_rel(mx / k - origin[0], my / k - origin[1])
                if 0 <= rx <= 1 and 0 <= ry <= 1:
                    cv2.drawMarker(view, (mx, my), (0, 255, 255), cv2.MARKER_CROSS, 16, 1)
                    _text(view, f"rel ({rx:.3f}, {ry:.3f})", (10, view.shape[0] - 10))
            cv2.imshow(WIN_NAME, view)
            key = cv2.waitKey(1) & 0xFF
            if key in QUIT_KEYS:
                break
            if key == ord("s"):
                out = Path(cfg.paths.debug)
                out.mkdir(parents=True, exist_ok=True)
                p = out / f"overlay-{time.strftime('%Y%m%d-%H%M%S')}.png"
                cv2.imwrite(str(p), frame)
                print(f"saved {p}")
    finally:
        cv2.destroyAllWindows()
        grabber.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
