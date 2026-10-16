"""Safety primitives: tap rate limit, corner failsafe, global pause/quit hotkeys."""
from __future__ import annotations

import threading
import time
from typing import Callable, List, Optional

from geometry import Rect, in_corner


class FailsafeTriggered(RuntimeError):
    """Cursor was moved into a screen corner — stop everything."""


class QuitRequested(RuntimeError):
    """The quit hotkey was pressed."""


class RateLimiter:
    """Enforce a minimum interval between events (max_per_sec)."""

    def __init__(self, max_per_sec: float, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if max_per_sec <= 0:
            raise ValueError("max_per_sec must be > 0")
        self.min_interval = 1.0 / max_per_sec
        self._clock, self._sleep = clock, sleep
        self._last: Optional[float] = None

    def wait(self) -> float:
        """Block until the next event is allowed; return seconds slept."""
        now = self._clock()
        slept = 0.0
        if self._last is not None:
            remaining = self.min_interval - (now - self._last)
            if remaining > 0:
                self._sleep(remaining)
                slept = remaining
        self._last = self._clock()
        return slept


# --------------------------------------------------------------------------- failsafe
def display_rects() -> List[Rect]:
    import Quartz  # lazy

    err, ids, count = Quartz.CGGetActiveDisplayList(16, None, None)
    rects = []
    for did in (ids or [])[:count]:
        b = Quartz.CGDisplayBounds(did)
        rects.append(Rect(b.origin.x, b.origin.y, b.size.width, b.size.height))
    return rects


def cursor_position() -> tuple:
    import Quartz  # lazy

    loc = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
    return (loc.x, loc.y)


def check_failsafe(margin: float,
                   cursor: Callable[[], tuple] = cursor_position,
                   displays: Callable[[], List[Rect]] = display_rects) -> None:
    """Raise FailsafeTriggered if the cursor is parked in any display corner."""
    x, y = cursor()
    if in_corner(x, y, displays(), margin):
        raise FailsafeTriggered(f"cursor in screen corner at ({x:.0f}, {y:.0f}) — failsafe stop")


# --------------------------------------------------------------------------- hotkeys
class Hotkeys:
    """Global pause/resume + quit hotkeys via pynput (needs Accessibility/Input Monitoring)."""

    def __init__(self, pause_combo: str, quit_combo: str) -> None:
        self.pause_combo, self.quit_combo = pause_combo, quit_combo
        self._paused = threading.Event()
        self._quit = threading.Event()
        self._listener = None

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    @property
    def quit_requested(self) -> bool:
        return self._quit.is_set()

    def toggle_pause(self) -> None:
        if self._paused.is_set():
            self._paused.clear()
            print("[hotkey] resumed", flush=True)
        else:
            self._paused.set()
            print("[hotkey] paused", flush=True)

    def request_quit(self) -> None:
        self._quit.set()
        self._paused.clear()
        print("[hotkey] quit requested", flush=True)

    def start(self) -> bool:
        """Start listening. Returns False (and keeps running without hotkeys) if unavailable."""
        try:
            from pynput import keyboard  # lazy
        except Exception as e:  # pragma: no cover - environment dependent
            print(f"[hotkeys] unavailable ({e}); use Ctrl+C or the corner failsafe to stop")
            return False
        self._listener = keyboard.GlobalHotKeys({
            self.pause_combo: self.toggle_pause,
            self.quit_combo: self.request_quit,
        })
        self._listener.daemon = True
        self._listener.start()
        print(f"[hotkeys] pause/resume: {self.pause_combo}   quit: {self.quit_combo}")
        return True

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def checkpoint(self, poll_s: float = 0.1, on_wait: Optional[Callable[[], None]] = None) -> None:
        """Raise QuitRequested if quitting; block while paused (calling on_wait periodically)."""
        while self._paused.is_set() and not self._quit.is_set():
            if on_wait:
                on_wait()
            time.sleep(poll_s)
        if self._quit.is_set():
            raise QuitRequested("quit hotkey pressed")
