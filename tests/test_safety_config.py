import numpy as np
import pytest

from actions import Tapper
from capture import Rect, ScreenGeometry, WindowTracker, wait_for_stable
from config import Config, ConfigError, from_dict, load_config
from safety import Failsafe, FailsafeTriggered, RateLimiter, hotkey_pressed, in_corner, parse_hotkey


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


# --- config ---------------------------------------------------------------------------------

def test_repo_config_loads():
    cfg = load_config()
    assert cfg.window.owner_name == "iPhone Mirroring"
    assert cfg.screen.content == "auto"


def test_from_dict_overrides_and_coerces():
    cfg = from_dict({"clicks": {"max_taps_per_second": 2, "backend": "pyautogui"},
                     "screen": {"content": [0, 0.05, 1, 0.9]}})
    assert cfg.clicks.max_taps_per_second == 2.0 and isinstance(cfg.clicks.max_taps_per_second, float)
    assert cfg.clicks.backend == "pyautogui"
    assert cfg.screen.content == [0.0, 0.05, 1.0, 0.9]


@pytest.mark.parametrize("data", [
    {"clicks": {"backend": "xdotool"}},
    {"clicks": {"nope": 1}},
    {"bogus": {}},
    {"clicks": {"max_taps_per_second": "fast"}},
    {"clicks": {"activate_before_tap": 1}},
    {"screen": {"content": [0.5, 0.5, 0.8, 0.2]}},
    {"screen": {"content": "manual"}},
])
def test_invalid_config_rejected(data):
    with pytest.raises(ConfigError):
        from_dict(data)


# --- safety ---------------------------------------------------------------------------------

def test_in_corner_multi_display():
    displays = [Rect(0, 0, 1440, 900), Rect(1440, 0, 1920, 1080)]
    assert in_corner((0, 0), displays, 5)
    assert in_corner((1437, 897), displays, 5)
    assert in_corner((3360, 1080), displays, 5)
    assert not in_corner((700, 450), displays, 5)
    assert not in_corner((1440, 500), displays, 5)


def test_failsafe_raises():
    fs = Failsafe(5, cursor=lambda: (1, 2), displays=lambda: [Rect(0, 0, 100, 100)])
    with pytest.raises(FailsafeTriggered):
        fs.check()
    Failsafe(5, cursor=lambda: (50, 50), displays=lambda: [Rect(0, 0, 100, 100)]).check()


def test_rate_limiter_spaces_actions():
    clk = FakeClock()
    rl = RateLimiter(4, clock=clk, sleep=clk.sleep)
    assert rl.wait() == 0
    assert rl.wait() == pytest.approx(0.25)
    clk.t += 1.0
    assert rl.wait() == 0
    with pytest.raises(ValueError):
        RateLimiter(0)


def test_parse_hotkey():
    assert parse_hotkey("ctrl+alt+p") == (35, 0x40000 | 0x80000)
    assert parse_hotkey("F8") == (100, 0)
    with pytest.raises(ValueError):
        parse_hotkey("ctrl+hyper+p")
    with pytest.raises(ValueError):
        parse_hotkey("ctrl+")


def test_hotkey_pressed_requires_exact_modifiers():
    combo = parse_hotkey("ctrl+alt+q")
    down = lambda code: code == 12  # noqa: E731
    assert hotkey_pressed(combo, down, 0x40000 | 0x80000 | 0x100)  # non-modifier bits ignored
    assert not hotkey_pressed(combo, down, 0x40000)
    assert not hotkey_pressed(combo, down, 0x40000 | 0x80000 | 0x20000)
    assert not hotkey_pressed(combo, lambda c: False, 0x40000 | 0x80000)


# --- tracker / stability / tapper -----------------------------------------------------------

GEO = ScreenGeometry(Rect(0, 0, 400, 900), Rect(10, 40, 380, 820), 2.0)


def test_tracker_refreshes_every_interval():
    clk = FakeClock()
    calls = []
    tracker = WindowTracker(Config(), grabber=None, resolver=lambda: calls.append(1) or (None, GEO),
                            clock=clk)
    tracker.get()
    clk.t = 1.0
    tracker.get()
    assert len(calls) == 1
    clk.t = 2.5
    tracker.get()
    assert len(calls) == 2
    tracker.get(force=True)
    assert len(calls) == 3


def test_wait_for_stable():
    clk = FakeClock()
    a = np.zeros((50, 50, 3), np.uint8)
    b = np.full((50, 50, 3), 255, np.uint8)
    frames = iter([a, b, a, a])
    frame, stable = wait_for_stable(lambda: next(frames), 0.01, 5.0, 0.1, clock=clk, sleep=clk.sleep)
    assert stable and frame is a
    flicker = iter([a, b] * 50)
    _, stable = wait_for_stable(lambda: next(flicker), 0.01, 0.35, 0.1, clock=clk, sleep=clk.sleep)
    assert not stable


class _Tracker:
    window = None

    def get(self, force=False):
        return GEO


def _tapper(dry_run=False, cursor=(200, 200)):
    clicks = []
    clk = FakeClock()
    t = Tapper(Config(), _Tracker(), dry_run=dry_run,
               clicker=type("C", (), {"click": lambda self, x, y: clicks.append((x, y))})(),
               failsafe=Failsafe(5, cursor=lambda: cursor, displays=lambda: [Rect(0, 0, 1440, 900)]),
               limiter=RateLimiter(4, clock=clk, sleep=clk.sleep), activator=lambda: None,
               log=lambda s: None)
    return t, clicks


def test_tapper_clicks_converted_point():
    t, clicks = _tapper()
    r = t.tap(0.5, 0.5)
    assert r.executed and clicks == [(200.0, 450.0)]


def test_tapper_dry_run_never_clicks():
    t, clicks = _tapper(dry_run=True)
    assert not t.tap(0.1, 0.9).executed and clicks == []


def test_tapper_failsafe_blocks_click():
    t, clicks = _tapper(cursor=(0, 0))
    with pytest.raises(FailsafeTriggered):
        t.tap(0.5, 0.5)
    assert clicks == []
