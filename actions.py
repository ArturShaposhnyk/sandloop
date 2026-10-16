"""Tap the iPhone screen by clicking into the iPhone Mirroring window.

CLI:
  python -m actions tap 0.5 0.8                      # one tap at relative (0.5, 0.8)
  python -m actions tap 0.5 0.8 --backend pyautogui  # try the fallback click backend
  python -m actions tap 0.5 0.8 --dry-run            # resolve + print, never click
  python -m actions tap 0.5 0.8 --repeat 5           # 5 taps (rate-limited, hotkeys active)

Every tap goes through `Tapper.tap`, which enforces: quit/pause hotkeys, the corner failsafe,
relative bounds 0..1, "inside the detected iPhone screen rect", the taps/sec rate limit,
dry-run, and window activation. Pixel/point conversion comes from geometry.ScreenGeometry.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from capture import CaptureError, WindowTracker
from config import Config, load_config
from geometry import ScreenGeometry, is_valid_rel
from safety import FailsafeTriggered, Hotkeys, QuitRequested, RateLimiter, check_failsafe


class TapRejected(ValueError):
    """Tap target failed a safety check and was not sent."""


# --------------------------------------------------------------------------- click backends
def click_quartz(x: float, y: float, press_s: float) -> None:
    import Quartz  # lazy

    pt = Quartz.CGPointMake(x, y)
    tap = Quartz.kCGHIDEventTap
    move = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventMouseMoved, pt, Quartz.kCGMouseButtonLeft)
    down = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventLeftMouseDown, pt, Quartz.kCGMouseButtonLeft)
    up = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventLeftMouseUp, pt, Quartz.kCGMouseButtonLeft)
    Quartz.CGEventPost(tap, move)
    time.sleep(0.01)
    Quartz.CGEventPost(tap, down)
    time.sleep(press_s)
    Quartz.CGEventPost(tap, up)


def click_pyautogui(x: float, y: float, press_s: float) -> None:
    import pyautogui  # lazy

    pyautogui.FAILSAFE = True  # mouse in a corner raises pyautogui.FailSafeException
    pyautogui.PAUSE = 0
    pyautogui.moveTo(x, y)
    pyautogui.mouseDown()
    time.sleep(press_s)
    pyautogui.mouseUp()


CLICKERS = {"quartz": click_quartz, "pyautogui": click_pyautogui}


# --------------------------------------------------------------------------- window focus
def frontmost_pid() -> Optional[int]:
    try:
        from AppKit import NSWorkspace  # lazy; ships with pyobjc-framework-Cocoa
    except ImportError:
        return None
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    return int(app.processIdentifier()) if app is not None else None


def activate_pid(pid: int) -> bool:
    """Bring the process to front via osascript (by PID, so localized app names don't matter)."""
    script = (f'tell application "System Events" to set frontmost of '
              f'(first process whose unix id is {int(pid)}) to true')
    res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=5)
    if res.returncode != 0:
        print(f"[actions] activation failed: {res.stderr.strip()} — grant Accessibility / "
              "Automation (System Events) to your terminal", file=sys.stderr)
        return False
    return True


# --------------------------------------------------------------------------- tapper
@dataclass
class TapResult:
    rel: Tuple[float, float]
    point: Tuple[float, float]
    sent: bool  # False in dry-run


class Tapper:
    def __init__(self, cfg: Config, tracker: WindowTracker, hotkeys: Optional[Hotkeys] = None,
                 clicker: Optional[Callable[[float, float, float], None]] = None,
                 failsafe: Optional[Callable[[float], None]] = None,
                 activator: Optional[Callable[[], None]] = None) -> None:
        self.cfg, self.tracker, self.hotkeys = cfg, tracker, hotkeys
        self.clicker = clicker or CLICKERS[cfg.click.backend]
        self.failsafe = failsafe or check_failsafe
        self.activator = activator or self._ensure_front
        self.limiter = RateLimiter(cfg.safety.max_taps_per_sec)

    def _ensure_front(self) -> None:
        if not self.cfg.click.activate_before_click:
            return
        pid = self.tracker.window.pid
        if frontmost_pid() != pid and activate_pid(pid):
            time.sleep(self.cfg.click.activate_settle_s)

    def resolve(self, rx: float, ry: float, geom: Optional[ScreenGeometry] = None) -> Tuple[float, float]:
        """Validate a relative target and return its global point. Raises TapRejected."""
        if not is_valid_rel(rx, ry):
            raise TapRejected(f"relative target ({rx}, {ry}) is outside 0..1")
        g = geom or self.tracker.geometry()
        x, y = g.rel_to_point(rx, ry)
        if not g.contains_point(x, y):
            raise TapRejected(f"point ({x:.1f}, {y:.1f}) is outside the iPhone screen rect {g.content_pt}")
        return (x, y)

    def tap(self, rx: float, ry: float) -> TapResult:
        if self.hotkeys is not None:
            self.hotkeys.checkpoint()
        self.failsafe(self.cfg.safety.failsafe_corner_pt)
        x, y = self.resolve(rx, ry)
        self.limiter.wait()
        if self.cfg.safety.dry_run:
            return TapResult((rx, ry), (x, y), sent=False)
        self.activator()
        # Re-check after activation/rate-limit sleep: window may have moved, cursor may be in a corner.
        self.failsafe(self.cfg.safety.failsafe_corner_pt)
        x, y = self.resolve(rx, ry)
        self.clicker(x, y, self.cfg.click.press_s)
        return TapResult((rx, ry), (x, y), sent=True)


# --------------------------------------------------------------------------- CLI
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m actions", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tap", help="tap a relative point of the iPhone screen")
    p.add_argument("rx", type=float)
    p.add_argument("ry", type=float)
    p.add_argument("--backend", choices=tuple(CLICKERS), help="override click.backend")
    p.add_argument("--dry-run", action="store_true", help="resolve and print, never click")
    p.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.backend:
        cfg.click.backend = args.backend
    if args.dry_run:
        cfg.safety.dry_run = True

    hotkeys = Hotkeys(cfg.safety.hotkey_pause, cfg.safety.hotkey_quit) if args.repeat > 1 else None
    if hotkeys:
        hotkeys.start()
    try:
        tapper = Tapper(cfg, WindowTracker(cfg), hotkeys=hotkeys)
        for i in range(max(1, args.repeat)):
            r = tapper.tap(args.rx, args.ry)
            verb = "tapped" if r.sent else "[dry-run] would tap"
            print(f"{verb} rel=({r.rel[0]:.3f}, {r.rel[1]:.3f}) -> point=({r.point[0]:.1f}, "
                  f"{r.point[1]:.1f}) via {cfg.click.backend}  [{i + 1}/{args.repeat}]")
    except (CaptureError, TapRejected) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except (FailsafeTriggered, QuitRequested) as e:
        print(f"stopped: {e}", file=sys.stderr)
        return 3
    except Exception as e:  # pyautogui.FailSafeException and friends
        if type(e).__name__ == "FailSafeException":
            print(f"stopped: pyautogui failsafe ({e})", file=sys.stderr)
            return 3
        raise
    finally:
        if hotkeys:
            hotkeys.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
