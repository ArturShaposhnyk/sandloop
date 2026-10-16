import numpy as np
import pytest

from config import StabilityConfig
from geometry import Rect
from safety import FailsafeTriggered, Hotkeys, QuitRequested, RateLimiter, check_failsafe
from stability import frame_diff, wait_stable


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


def test_rate_limiter_spacing():
    clk = FakeClock()
    rl = RateLimiter(4.0, clock=clk, sleep=clk.sleep)
    assert rl.wait() == 0.0
    assert rl.wait() == pytest.approx(0.25)
    clk.t += 1.0
    assert rl.wait() == 0.0
    clk.t += 0.1
    assert rl.wait() == pytest.approx(0.15)


def test_rate_limiter_rejects_zero():
    with pytest.raises(ValueError):
        RateLimiter(0)


def test_failsafe():
    displays = lambda: [Rect(0, 0, 1728, 1117)]  # noqa: E731
    check_failsafe(5, cursor=lambda: (800, 500), displays=displays)
    with pytest.raises(FailsafeTriggered):
        check_failsafe(5, cursor=lambda: (1, 1116), displays=displays)


def test_hotkeys_state_machine():
    hk = Hotkeys("<ctrl>+<alt>+p", "<ctrl>+<alt>+q")
    hk.checkpoint()  # not paused, no quit: returns immediately
    hk.toggle_pause()
    assert hk.paused
    calls = []
    hk_wait = lambda: (calls.append(1), hk.toggle_pause())  # noqa: E731 - resume on first wait
    hk.checkpoint(poll_s=0, on_wait=hk_wait)
    assert calls and not hk.paused
    hk.request_quit()
    with pytest.raises(QuitRequested):
        hk.checkpoint()


def _frame(v: int) -> np.ndarray:
    return np.full((200, 100, 3), v, dtype=np.uint8)


def test_frame_diff():
    assert frame_diff(_frame(10), _frame(10)) == 0.0
    assert frame_diff(_frame(10), _frame(30)) == pytest.approx(20.0)
    assert frame_diff(_frame(10), np.zeros((50, 50, 3), np.uint8)) == float("inf")


def test_wait_stable_settles():
    frames = iter([_frame(0), _frame(50), _frame(100), _frame(101), _frame(101), _frame(101)])
    clk = FakeClock()
    cfg = StabilityConfig(threshold=2.0, consecutive=2, timeout_s=5.0, poll_s=0.1)
    res = wait_stable(lambda: next(frames), cfg, clock=clk, sleep=clk.sleep)
    assert res.stable
    assert int(res.frame[0, 0, 0]) == 101


def test_wait_stable_times_out():
    counter = {"v": 0}

    def grab():
        counter["v"] = (counter["v"] + 40) % 256
        return _frame(counter["v"])

    clk = FakeClock()
    cfg = StabilityConfig(threshold=2.0, consecutive=2, timeout_s=1.0, poll_s=0.1)
    res = wait_stable(grab, cfg, clock=clk, sleep=clk.sleep)
    assert not res.stable
    assert res.waited_s >= 1.0
