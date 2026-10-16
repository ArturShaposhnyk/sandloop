"""Live debug view of the iPhone screen with FPS and the relative coordinate under the cursor.

    python -m tools.overlay            # iPhone screen area only
    python -m tools.overlay --window   # whole window, detected screen rect drawn in green
    python -m tools.overlay --dry-run  # + read stable gameplay frames, draw the chosen move

--dry-run never clicks (this tool does not import actions): it shows the grid detections and the
greedy strategy's next move (cyan ring "TAP"; rejected candidates crossed out in orange).

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
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Optional, Sequence, Tuple

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


class DryRun:
    """Runs ``analyze(frame)`` (strategy.Analyzer.analyze) on a worker thread for stable frames,
    at most every ``interval_s``, and draws the latest result onto later frames. Never taps."""

    def __init__(self, analyze, interval_s: float, stable_below: float,
                 clock=time.monotonic) -> None:
        self._analyze = analyze
        self.interval_s = interval_s
        self.stable_below = stable_below
        self._clock = clock
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._future: Optional[Future] = None
        self._last = -float("inf")
        self.result: Any = None          # strategy.Analysis
        self.result_shape: Optional[Tuple[int, ...]] = None
        self.result_time: Optional[float] = None
        self.last_reason: Optional[str] = None

    def feed(self, content: np.ndarray, diff: float) -> None:
        if self._future is not None and self._future.done():
            try:
                res, shape = self._future.result()
                self.result, self.result_shape, self.result_time = res, shape, self._clock()
            except Exception as e:  # keep the live view running; show the error instead
                self.result, self.result_shape = None, None
                print(f"dry-run: analysis failed: {e!r}")
            self._future = None
            self._report()
        now = self._clock()
        if self._future is None and diff < self.stable_below and now - self._last >= self.interval_s:
            self._last = now
            frame = content.copy()
            self._future = self._pool.submit(
                lambda: (self._analyze(frame, log=lambda s: print("dry-run: " + s)), frame.shape))

    def _report(self) -> None:
        res = self.result
        reason = None if res is None else (
            res.decision.reason if res.decision is not None else f"screen: {res.screen}")
        if reason != self.last_reason:
            tap = res.tap_rel() if res is not None else None
            print(f"dry-run: {reason}" + (f" at rel ({tap[0]:.3f}, {tap[1]:.3f}) — not clicked"
                                          if tap else ""))
            self.last_reason = reason

    def status(self) -> str:
        res = self.result
        if res is None:
            return "dry-run: analysing..." if self._future is not None else "dry-run: waiting for a stable frame"
        age = self._clock() - (self.result_time or self._clock())
        if res.decision is None:
            return f"dry-run: {res.screen}{f' ({res.error})' if res.error else ''}"
        return f"dry-run ({age:.1f}s ago): {res.decision.reason}"

    def annotate(self, content: np.ndarray) -> np.ndarray:
        """Draw the latest gameplay decision onto ``content`` (same-size frames only)."""
        res = self.result
        if res is None or res.decision is None or self.result_shape != content.shape:
            return content
        from strategy import draw_decision

        return draw_decision(content, res.view, res.decision)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import cv2

    import permissions
    from safety import Hotkeys

    ap = argparse.ArgumentParser(prog="python -m tools.overlay", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", action="store_true", help="show the whole window + screen rect")
    ap.add_argument("--dry-run", action="store_true",
                    help="analyse gameplay frames and draw the chosen move (never clicks)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    permissions.require(screen=True, accessibility=False)
    dry: Optional[DryRun] = None
    if args.dry_run:
        from strategy import Analyzer

        dry = DryRun(Analyzer(cfg).analyze, cfg.strategy.analyze_interval_s,
                     cfg.stability.threshold)

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
            diff = frame_change(prev, frame) if prev is not None and prev.shape == frame.shape else 1.0
            prev = frame
            if dry is not None:
                if args.window:   # analyse and annotate only the iPhone screen inside the window
                    x0, y0 = int(round(origin[0])), int(round(origin[1]))
                    cw, chh = geo.frame_size
                    shown = frame.copy()
                    sub = shown[y0:y0 + chh, x0:x0 + cw]
                    if not hotkeys.paused.is_set():
                        dry.feed(sub, diff)
                    shown[y0:y0 + chh, x0:x0 + cw] = dry.annotate(sub)
                else:
                    if not hotkeys.paused.is_set():
                        dry.feed(frame, diff)
                    shown = dry.annotate(frame)
            else:
                shown = frame
            k = min(1.0, cfg.overlay.max_height / frame.shape[0])
            view = cv2.resize(shown, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else shown.copy()
            _text(view, f"FPS {fps.tick():5.1f}  diff {diff:.4f}  scale {geo.scale:.2f}", (10, 25))
            _text(view, f"screen {geo.content.w:.0f}x{geo.content.h:.0f}pt @ "
                        f"({geo.content.x:.0f},{geo.content.y:.0f})", (10, 50))
            if dry is not None:
                _text(view, dry.status()[:70], (10, 75))
            if hotkeys.paused.is_set():
                _text(view, "PAUSED", (10, 100))
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
        if dry is not None:
            dry.close()
        cv2.destroyAllWindows()
        grabber.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
