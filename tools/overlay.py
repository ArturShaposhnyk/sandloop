"""Live debug view of the iPhone screen with FPS and the relative coordinate under the cursor.

    python -m tools.overlay            # iPhone screen area only
    python -m tools.overlay --window   # whole window, detected screen rect drawn in green
    python -m tools.overlay --dry-run  # + read stable gameplay frames, draw the chosen move

--dry-run never clicks (this tool does not import actions): it shows the grid detections and the
greedy strategy's next move (cyan ring "TAP"; rejected candidates crossed out in orange).

Keys (overlay window focused): q / Esc quit, s save a report to debug/.
Global hotkeys from config.yaml work from any app (e.g. while playing in iPhone Mirroring):
pause / quit, and save (default ctrl+alt+s). A report is the raw iPhone screen, and with
--dry-run also the annotated frame and the decision as text (debug/report-<time>*).
If the mirroring window is missing (or disappears), a placeholder is shown until it comes back.
With screen.capture=mss keep the overlay away from the mirroring window (it would capture itself).
"""
from __future__ import annotations

import argparse
import sys
import threading
import textwrap
import time
from pathlib import Path
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, List, Optional, Sequence, Tuple

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
    at most every ``interval_s``, and draws the latest result onto later frames. Every result
    goes through a DecisionGate: a move counts as confirmed after ``confirm_frames`` agreeing
    analyses (what the bot will require before a real tap). Never taps."""

    def __init__(self, analyze, interval_s: float, stable_below: float,
                 clock=time.monotonic, confirm_frames: int = 2) -> None:
        from strategy import DecisionGate

        self._analyze = analyze
        self.gate = DecisionGate(confirm_frames)
        self.confirmed: Any = None       # strategy.Candidate confirmed by the gate, or None
        self.interval_s = interval_s
        self.stable_below = stable_below
        self._clock = clock
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._future: Optional[Future] = None
        self._last = -float("inf")
        self.result: Any = None          # strategy.Analysis
        self.result_shape: Optional[Tuple[int, ...]] = None
        self.result_frame: Optional[np.ndarray] = None   # the frame the result is about
        self.result_time: Optional[float] = None
        self.last_reason: Optional[str] = None

    def feed(self, content: np.ndarray, diff: float) -> None:
        if self._future is not None and self._future.done():
            try:
                res, frame = self._future.result()
                self.result, self.result_shape, self.result_time = res, frame.shape, self._clock()
                self.result_frame = frame
                self.confirmed = self.gate.feed(res)
            except Exception as e:  # keep the live view running; show the error instead
                self.result, self.result_shape, self.result_frame = None, None, None
                self.gate.reset()
                self.confirmed = None
                print(f"dry-run: analysis failed: {e!r}")
            self._future = None
            self._report()
        now = self._clock()
        if self._future is None and diff < self.stable_below and now - self._last >= self.interval_s:
            self._last = now
            frame = content.copy()
            self._future = self._pool.submit(
                lambda: (self._analyze(frame, log=lambda s: print("dry-run: " + s), diff=diff),
                         frame))

    def _reason(self) -> Optional[str]:
        res = self.result
        if res is None:
            return None
        if res.failure:
            return f"read failed ({res.failure}) - wait"
        if res.decision is None:
            return f"screen: {res.screen}"
        if res.decision.move is None:
            return res.decision.reason
        if self.confirmed is not None:
            return f"{res.decision.reason} CONFIRMED {self.gate.frames}/{self.gate.frames}"
        return f"{res.decision.reason} pending {self.gate.count}/{self.gate.frames}"

    def _report(self) -> None:
        reason = self._reason()
        if reason is not None and "pending" in reason:
            return   # print decisions once confirmed (or waits / failures), not every attempt
        if reason != self.last_reason:
            tap = self.result.tap_rel() if self.confirmed is not None else None
            print(f"dry-run: {reason}" + (f" at rel ({tap[0]:.3f}, {tap[1]:.3f}) — not clicked"
                                          if tap else ""))
            self.last_reason = reason

    def fail_rate(self) -> str:
        res = self.result
        if res is None or not res.reads:
            return "read fails -"
        return f"read fails {res.fails}/{res.reads} ({res.fails / res.reads:.0%})"

    def status(self) -> str:
        res = self.result
        if res is None:
            return "dry-run: analysing..." if self._future is not None else "dry-run: waiting for a stable frame"
        age = self._clock() - (self.result_time or self._clock())
        return f"dry-run ({age:.1f}s ago): {self._reason()} | {self.fail_rate()}"

    def annotate(self, content: np.ndarray) -> np.ndarray:
        """Draw the latest gameplay decision onto ``content`` (same-size frames only)."""
        res = self.result
        if res is None or res.decision is None or self.result_shape != content.shape:
            return content
        from strategy import draw_decision

        return draw_decision(content, res.view, res.decision)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def save_report(out_dir: Path, raw: np.ndarray, dry: Optional[DryRun] = None,
                stamp: Optional[str] = None) -> List[Path]:
    """Save the raw iPhone screen now, and with a dry-run result also the frame that result was
    computed on (``-analysed``; it can be a few frames older), that frame annotated, and the
    decision as text. Raises OSError if an image can't be written."""
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    base = out_dir / f"report-{stamp}"
    saved = []

    def write(path: Path, img: np.ndarray) -> None:
        if not cv2.imwrite(str(path), img):
            raise OSError(f"cannot write {path}")
        saved.append(path)

    write(base.with_suffix(".png"), raw)
    if dry is not None:
        lines = [dry.status()]
        res = dry.result
        if dry.result_frame is not None:
            write(Path(f"{base}-analysed.png"), dry.result_frame)
        if res is not None and res.decision is not None:
            from strategy import draw_decision

            write(Path(f"{base}-annotated.png"),
                  draw_decision(dry.result_frame, res.view, res.decision))
            tap = res.tap_rel()
            lines += [f"screen: {res.screen}", f"decision: {res.decision.reason}"
                      + (f" at rel ({tap[0]:.3f}, {tap[1]:.3f})" if tap else "")]
            lines += ["grid:"] + ["  " + row for row in res.view.labels()]
            lines += res.state.summary()
            lines += ["candidates:"] + ["  " + c.describe() for c in res.decision.candidates]
            lines += [f"sand: {res.state.sand}", f"exposed: {res.state.exposed}"]
        elif res is not None:
            lines.append(f"screen: {res.screen}" + (f" ({res.error})" if res.error else ""))
        txt = Path(f"{base}.txt")
        txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
        saved.append(txt)
    return saved


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
                     cfg.stability.threshold, confirm_frames=cfg.strategy.confirm_frames)

    grabber = Grabber(cfg.screen.capture)
    tracker = WindowTracker(cfg, grabber)
    save_req = threading.Event()   # global hotkey: works while iPhone Mirroring has the focus
    hotkeys = Hotkeys(cfg.safety.hotkey_pause, cfg.safety.hotkey_quit,
                      extra={cfg.overlay.hotkey_save: lambda: (print("[hotkey] save"),
                                                               save_req.set())}).start()
    fps = Fps()
    mouse = {"pos": None}  # display-space cursor position
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN_NAME, lambda ev, x, y, *_: mouse.__setitem__("pos", (x, y)))
    prev: Optional[np.ndarray] = None
    frame: Optional[np.ndarray] = None
    print(f"overlay running — q/Esc to quit, s to save (overlay focused); hotkeys: "
          f"save={cfg.overlay.hotkey_save} pause={hotkeys.pause_spec} quit={hotkeys.quit_spec}")
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
            raw = frame
            if dry is not None:
                if args.window:   # analyse and annotate only the iPhone screen inside the window
                    x0, y0 = int(round(origin[0])), int(round(origin[1]))
                    cw, chh = geo.frame_size
                    shown = frame.copy()
                    sub = shown[y0:y0 + chh, x0:x0 + cw]
                    raw = frame[y0:y0 + chh, x0:x0 + cw]
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
                st = dry.status()
                _text(view, st[:75], (10, 75))
                if len(st) > 75:
                    _text(view, st[75:150], (10, 100))
            if hotkeys.paused.is_set():
                _text(view, "PAUSED", (10, 125))
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
            if key == ord("s") or save_req.is_set():
                save_req.clear()
                try:
                    paths = save_report(Path(cfg.paths.debug), raw, dry)
                    print("saved " + ", ".join(str(p) for p in paths))
                except OSError as e:
                    print(f"save failed: {e}")
    finally:
        if dry is not None:
            dry.close()
        cv2.destroyAllWindows()
        grabber.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
