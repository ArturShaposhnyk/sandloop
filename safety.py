"""Safety: corner failsafe, tap rate limit, global pause/quit hotkeys.

Pure parts (corner check, rate limiter, hotkey parsing) are testable anywhere; cursor/keyboard
polling uses Quartz, imported lazily.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from capture import Rect


class FailsafeTriggered(RuntimeError):
    """Raised when the cursor is parked in a screen corner (the user wants the bot to stop)."""


class QuitRequested(RuntimeError):
    pass


# --- failsafe ---------------------------------------------------------------------------------


def in_corner(pos: Tuple[float, float], displays: Sequence[Rect], margin: float) -> bool:
    x, y = pos
    for d in displays:
        for cx, cy in ((d.x, d.y), (d.right, d.y), (d.x, d.bottom), (d.right, d.bottom)):
            if abs(x - cx) <= margin and abs(y - cy) <= margin:
                return True
    return False


def cursor_position() -> Tuple[float, float]:
    import Quartz

    loc = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
    return float(loc.x), float(loc.y)


def display_bounds() -> List[Rect]:
    import Quartz

    err, ids, _count = Quartz.CGGetActiveDisplayList(16, None, None)
    if err:
        ids = [Quartz.CGMainDisplayID()]
    out = []
    for i in ids:
        b = Quartz.CGDisplayBounds(i)
        out.append(Rect(b.origin.x, b.origin.y, b.size.width, b.size.height))
    return out


class Failsafe:
    def __init__(self, margin: float,
                 cursor: Callable[[], Tuple[float, float]] = cursor_position,
                 displays: Callable[[], List[Rect]] = display_bounds) -> None:
        self.margin = margin
        self._cursor = cursor
        self._displays = displays

    def check(self) -> None:
        pos = self._cursor()
        if in_corner(pos, self._displays(), self.margin):
            raise FailsafeTriggered(f"cursor in a screen corner at {pos} — stopping")


# --- rate limit -------------------------------------------------------------------------------


class RateLimiter:
    """Enforces a minimum interval between actions (max N per second)."""

    def __init__(self, max_per_second: float, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if max_per_second <= 0:
            raise ValueError("max_per_second must be > 0")
        self.min_interval = 1.0 / max_per_second
        self._clock = clock
        self._sleep = sleep
        self._last: Optional[float] = None

    def wait(self) -> float:
        """Block until the next action is allowed; returns the time slept."""
        now = self._clock()
        slept = 0.0
        if self._last is not None:
            slept = max(0.0, self._last + self.min_interval - now)
            if slept > 0:
                self._sleep(slept)
                now = self._clock()
        self._last = now
        return slept


# --- hotkeys ----------------------------------------------------------------------------------

# macOS virtual key codes (kVK_*), US layout positions.
KEYCODES = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9, "b": 11,
    "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17, "1": 18, "2": 19, "3": 20, "4": 21,
    "6": 22, "5": 23, "9": 25, "7": 26, "8": 28, "0": 29, "o": 31, "u": 32, "i": 34, "p": 35,
    "l": 37, "j": 38, "k": 40, "n": 45, "m": 46, "tab": 48, "space": 49, "esc": 53,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97, "f7": 98, "f8": 100,
    "f9": 101, "f10": 109, "f11": 103, "f12": 111,
}
MODIFIERS = {  # CGEventFlags masks
    "shift": 0x20000, "ctrl": 0x40000, "control": 0x40000,
    "alt": 0x80000, "opt": 0x80000, "option": 0x80000, "cmd": 0x100000, "command": 0x100000,
}
_MOD_MASK = 0x20000 | 0x40000 | 0x80000 | 0x100000


def parse_hotkey(spec: str) -> Tuple[int, int]:
    """'ctrl+alt+p' -> (keycode, modifier flags)."""
    parts = [p.strip().lower() for p in spec.split("+") if p.strip()]
    if not parts:
        raise ValueError(f"empty hotkey: {spec!r}")
    *mods, key = parts
    if key not in KEYCODES:
        raise ValueError(f"unknown key {key!r} in hotkey {spec!r}")
    flags = 0
    for m in mods:
        if m not in MODIFIERS:
            raise ValueError(f"unknown modifier {m!r} in hotkey {spec!r}")
        flags |= MODIFIERS[m]
    return KEYCODES[key], flags


def hotkey_pressed(combo: Tuple[int, int], key_down: Callable[[int], bool], flags: int) -> bool:
    keycode, mods = combo
    return key_down(keycode) and (flags & _MOD_MASK) == mods


class Hotkeys:
    """Polls the keyboard state in a background thread (no event tap needed).

    Pause hotkey toggles ``paused``; quit hotkey sets ``quit``. ``extra`` maps more hotkey specs
    to callbacks, called once per press (on key down) from the polling thread.
    """

    def __init__(self, pause_spec: str, quit_spec: str, poll_s: float = 0.05,
                 extra: Optional[Dict[str, Callable[[], None]]] = None) -> None:
        self.pause_combo = parse_hotkey(pause_spec)
        self.quit_combo = parse_hotkey(quit_spec)
        self.pause_spec, self.quit_spec = pause_spec, quit_spec
        self.paused = threading.Event()
        self.quit = threading.Event()
        self._extra = [(parse_hotkey(spec), cb) for spec, cb in (extra or {}).items()]
        self._poll_s = poll_s
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "Hotkeys":
        self._thread = threading.Thread(target=self._run, name="hotkeys", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        import Quartz

        state = Quartz.kCGEventSourceStateCombinedSessionState
        key_down = lambda code: bool(Quartz.CGEventSourceKeyState(state, code))  # noqa: E731
        was_pause = False
        was_extra = [False] * len(self._extra)
        while not self.quit.is_set():
            flags = int(Quartz.CGEventSourceFlagsState(state))
            if hotkey_pressed(self.quit_combo, key_down, flags):
                self.quit.set()
                break
            is_pause = hotkey_pressed(self.pause_combo, key_down, flags)
            if is_pause and not was_pause:
                if self.paused.is_set():
                    self.paused.clear()
                    print("[hotkey] resumed")
                else:
                    self.paused.set()
                    print("[hotkey] paused")
            was_pause = is_pause
            for i, (combo, cb) in enumerate(self._extra):
                down = hotkey_pressed(combo, key_down, flags)
                if down and not was_extra[i]:
                    cb()
                was_extra[i] = down
            time.sleep(self._poll_s)
