"""Tap the iPhone screen by clicking into the iPhone Mirroring window.

All callers pass relative coordinates (0..1 of the iPhone screen); conversion to global points
happens here via ScreenGeometry.

CLI:
    python -m actions tap 0.5 0.8 [--dry-run] [--backend quartz|pyautogui] [--count N]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Sequence, Tuple

from capture import ScreenGeometry, WindowTracker
from config import Config, load_config
from safety import Failsafe, RateLimiter


class OutOfBounds(ValueError):
    pass


class Clicker(Protocol):
    def click(self, x: float, y: float) -> None: ...


class QuartzClicker:
    """CGEvent mouse move + down/up at global point coordinates."""

    def __init__(self, press_s: float, settle_s: float) -> None:
        import Quartz

        self._q = Quartz
        self.press_s, self.settle_s = press_s, settle_s

    def _post(self, kind: int, x: float, y: float) -> None:
        q = self._q
        ev = q.CGEventCreateMouseEvent(None, kind, (x, y), q.kCGMouseButtonLeft)
        q.CGEventPost(q.kCGHIDEventTap, ev)

    def click(self, x: float, y: float) -> None:
        q = self._q
        self._post(q.kCGEventMouseMoved, x, y)
        time.sleep(self.settle_s)
        self._post(q.kCGEventLeftMouseDown, x, y)
        time.sleep(self.press_s)
        self._post(q.kCGEventLeftMouseUp, x, y)


class PyAutoGuiClicker:
    def __init__(self, press_s: float, settle_s: float) -> None:
        import pyautogui

        pyautogui.FAILSAFE = True  # mouse in a corner raises pyautogui.FailSafeException
        pyautogui.PAUSE = 0
        self._p = pyautogui
        self.press_s, self.settle_s = press_s, settle_s

    def click(self, x: float, y: float) -> None:
        self._p.moveTo(x, y)
        time.sleep(self.settle_s)
        self._p.mouseDown(x, y)
        time.sleep(self.press_s)
        self._p.mouseUp(x, y)


def make_clicker(cfg: Config, backend: Optional[str] = None) -> Clicker:
    name = backend or cfg.clicks.backend
    cls = {"quartz": QuartzClicker, "pyautogui": PyAutoGuiClicker}.get(name)
    if cls is None:
        raise ValueError(f"unknown click backend {name!r}")
    return cls(cfg.clicks.press_duration_s, cfg.clicks.move_settle_s)


# --- window activation ------------------------------------------------------------------------


def frontmost_pid() -> Optional[int]:
    try:
        from AppKit import NSWorkspace
        from Foundation import NSDate, NSRunLoop
    except ImportError:
        return None
    # NSWorkspace only learns about app switches from notifications on the run loop; without a
    # run loop the answer stays stale (run 224631: "not frontmost" before every tap from 138 s
    # on while every tap worked)
    NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.01))
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    return int(app.processIdentifier()) if app is not None else None


def activate_app(bundle_id: str, owner_name: str) -> bool:
    for script in (f'tell application id "{bundle_id}" to activate',
                   f'tell application "{owner_name}" to activate'):
        r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        if r.returncode == 0:
            return True
    return False


# --- tapping ----------------------------------------------------------------------------------


@dataclass
class TapResult:
    rel: Tuple[float, float]
    point: Tuple[float, float]
    executed: bool


def target_point(geo: ScreenGeometry, rx: float, ry: float) -> Tuple[float, float]:
    """Relative -> global point, refusing anything outside the iPhone screen rect."""
    try:
        x, y = geo.rel_to_point(rx, ry)
    except ValueError as e:
        raise OutOfBounds(str(e)) from e
    if not geo.content.contains(x, y):
        raise OutOfBounds(f"point {(x, y)} outside the iPhone screen rect {geo.content}")
    return x, y


class Tapper:
    def __init__(self, cfg: Config, tracker: WindowTracker, dry_run: bool = False,
                 clicker: Optional[Clicker] = None, failsafe: Optional[Failsafe] = None,
                 limiter: Optional[RateLimiter] = None,
                 activator: Optional[Callable[[], None]] = None,
                 log: Callable[[str], None] = print) -> None:
        self.cfg = cfg
        self.tracker = tracker
        self.dry_run = dry_run
        self.clicker = clicker if clicker is not None or dry_run else make_clicker(cfg)
        self.failsafe = failsafe or Failsafe(cfg.safety.failsafe_corner_pt)
        self.limiter = limiter or RateLimiter(cfg.clicks.max_taps_per_second)
        self.activator = activator or self._ensure_focus
        self.log = log

    def _ensure_focus(self) -> None:
        if not self.cfg.clicks.activate_before_tap:
            return
        win = self.tracker.window
        if win is not None and frontmost_pid() == win.pid:
            return
        self.activate()

    def activate(self) -> bool:
        """Bring iPhone Mirroring to the front and wait until it is frontmost (clicks.focus_*):
        a click right after the activation may only focus the window (run 161536, TAP #1)."""
        if not activate_app(self.cfg.window.app_bundle_id, self.cfg.window.owner_name):
            self.log("warning: could not activate the iPhone Mirroring app via osascript")
            return False
        win = self.tracker.window
        end = time.monotonic() + self.cfg.clicks.focus_wait_s
        front = False
        while win is not None and not front and time.monotonic() < end:
            time.sleep(0.05)
            front = frontmost_pid() == win.pid
        if win is not None and not front:
            self.log(f"warning: iPhone Mirroring is not frontmost {self.cfg.clicks.focus_wait_s:.1f}s"
                     " after activating it")
        time.sleep(self.cfg.clicks.focus_settle_s)
        return front

    def prepare(self) -> None:
        """Before the first live tap: the window is frontmost and has settled."""
        if self.dry_run or not self.cfg.clicks.activate_before_tap:
            return
        try:
            self.tracker.get(force=True)
        except Exception as e:   # noqa: BLE001 - no window yet: the bot loop waits for it
            self.log(f"focus before the first tap: {e}")
            return
        win = self.tracker.window
        if win is not None and frontmost_pid() == win.pid:
            self.log("iPhone Mirroring is frontmost")
        elif self.activate():
            self.log("iPhone Mirroring activated before the first tap")

    def tap(self, rx: float, ry: float) -> TapResult:
        geo = self.tracker.get()
        x, y = target_point(geo, rx, ry)
        if self.dry_run:
            self.log(f"[dry-run] tap rel=({rx:.3f}, {ry:.3f}) -> point=({x:.1f}, {y:.1f})")
            return TapResult((rx, ry), (x, y), False)
        self.failsafe.check()
        self.limiter.wait()
        self.activator()
        geo = self.tracker.get()  # window may have moved while we waited
        x, y = target_point(geo, rx, ry)
        self.failsafe.check()
        self.clicker.click(x, y)
        self.log(f"tap rel=({rx:.3f}, {ry:.3f}) -> point=({x:.1f}, {y:.1f})")
        return TapResult((rx, ry), (x, y), True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m actions", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("tap", help="tap a relative point of the iPhone screen")
    t.add_argument("rx", type=float)
    t.add_argument("ry", type=float)
    t.add_argument("--dry-run", action="store_true", help="print the target, don't click")
    t.add_argument("--backend", choices=["quartz", "pyautogui"], default=None)
    t.add_argument("--count", type=int, default=1, help="number of taps (rate-limited)")
    t.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    import permissions
    from capture import CaptureError, Grabber, WindowNotFound

    cfg = load_config(args.config)
    permissions.require(screen=True, accessibility=not args.dry_run)
    grabber = Grabber(cfg.screen.capture)
    tracker = WindowTracker(cfg, grabber)
    try:
        tapper = Tapper(cfg, tracker, dry_run=args.dry_run,
                        clicker=None if args.dry_run else make_clicker(cfg, args.backend))
        for _ in range(max(1, args.count)):
            tapper.tap(args.rx, args.ry)
    except (WindowNotFound, CaptureError, OutOfBounds) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # FailsafeTriggered, pyautogui.FailSafeException
        if type(e).__name__ in ("FailsafeTriggered", "FailSafeException"):
            print(f"stopped by failsafe: {e}", file=sys.stderr)
            return 2
        raise
    finally:
        grabber.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
