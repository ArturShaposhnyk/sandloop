"""Capture/actions logic with the macOS parts stubbed out — no live game, no Quartz needed."""
import numpy as np
import pytest

import capture
from actions import TapRejected, Tapper
from capture import WindowInfo, WindowTracker, detect_content_rect, pick_window
from config import ContentDetectConfig, config_from_dict
from geometry import Rect
from safety import FailsafeTriggered

OPTS = ContentDetectConfig()


def _win(owner="iPhone Mirroring", bounds=Rect(100, 50, 400, 880), layer=0, wid=7, title=""):
    return WindowInfo(window_id=wid, owner=owner, title=title, pid=42, bounds=bounds, layer=layer)


def test_pick_window_matches_owner_case_insensitive():
    wins = [_win(owner="Code"), _win(owner="iphone mirroring", wid=9), _win(layer=25, wid=3)]
    assert pick_window(wins, "iPhone Mirroring").window_id == 9


def test_pick_window_prefers_largest_and_title_filter():
    wins = [_win(bounds=Rect(0, 0, 100, 100), wid=1, title="a"), _win(wid=2, title="b")]
    assert pick_window(wins, "iPhone Mirroring").window_id == 2
    assert pick_window(wins, "iPhone Mirroring", "a").window_id == 1
    assert pick_window(wins, "Bildschirm") is None


def _phone_frame(title_px=60, w=800, h=1760, chrome=(235, 235, 235)):
    img = np.empty((h, w, 3), np.uint8)
    img[:] = chrome
    rng = np.random.default_rng(0)
    img[title_px:] = rng.integers(0, 120, size=(h - title_px, w, 3), dtype=np.uint8)
    return img


def test_detect_content_rect_trims_title_bar():
    rect, reason = detect_content_rect(_phone_frame(), OPTS)
    assert rect.y == pytest.approx(60 / 1760)
    assert rect.h == pytest.approx(1700 / 1760)
    assert rect.x == 0 and rect.w == 1
    assert "detected" in reason


def test_detect_content_rect_full_window_when_no_chrome():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, size=(1760, 800, 3), dtype=np.uint8)
    rect, reason = detect_content_rect(img, OPTS)
    assert rect == Rect(0, 0, 1, 1)
    assert "full window" in reason


def test_detect_content_rect_rejects_bad_aspect():
    img = _phone_frame(title_px=700)  # leaves a wide area -> wrong aspect
    rect, reason = detect_content_rect(img, OPTS)
    assert rect == Rect(0, 0, 1, 1)
    assert "aspect" in reason


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def tracker(monkeypatch):
    state = {"win": _win()}
    monkeypatch.setattr(capture, "find_window", lambda cfg: state["win"])
    monkeypatch.setattr(capture, "grab_window", lambda win, backend: _phone_frame(
        w=int(win.bounds.w * 2), h=int(win.bounds.h * 2)))
    clk = FakeClock()
    tr = WindowTracker(config_from_dict({}), clock=clk)
    return tr, state, clk


def test_tracker_crop_and_scale(tracker):
    tr, _, _ = tracker
    crop, g = tr.grab()
    assert g.scale == 2.0
    assert crop.shape[:2] == (1700, 800)
    assert g.content_pt.y == pytest.approx(80)


def test_tracker_follows_moved_window(tracker):
    tr, state, clk = tracker
    tr.grab()
    state["win"] = _win(bounds=Rect(600, 200, 400, 880))
    clk.t = 1.0
    assert tr.geometry().window_pt.x == 100  # not refreshed yet (interval 2s)
    clk.t = 2.5
    assert tr.geometry().window_pt.x == 600
    assert tr.geometry().rel_to_point(0, 0) == pytest.approx((600, 230))


def test_manual_content_override(monkeypatch):
    monkeypatch.setattr(capture, "find_window", lambda cfg: _win())
    monkeypatch.setattr(capture, "grab_window", lambda win, backend: _phone_frame())
    tr = WindowTracker(config_from_dict({"window": {"content_rect": [0.1, 0.1, 0.8, 0.8]}}))
    _, g = tr.grab()
    assert g.content_frac == Rect(0.1, 0.1, 0.8, 0.8)
    assert tr.content_reason == "manual override from config"


def _tapper(tracker_tuple, **cfg_over):
    tr, _, _ = tracker_tuple
    clicks = []
    cfg = config_from_dict({"safety": {"max_taps_per_sec": 1000, **cfg_over}})
    t = Tapper(cfg, tr, clicker=lambda x, y, p: clicks.append((x, y)),
               failsafe=lambda m: None, activator=lambda: None)
    return t, clicks


def test_tap_sends_click_at_converted_point(tracker):
    t, clicks = _tapper(tracker)
    r = t.tap(0.5, 0.8)
    assert r.sent
    assert clicks == [pytest.approx((300, 80 + 0.8 * 850))]


def test_tap_rejects_out_of_range(tracker):
    t, clicks = _tapper(tracker)
    for rx, ry in [(-0.1, 0.5), (0.5, 1.2)]:
        with pytest.raises(TapRejected):
            t.tap(rx, ry)
    assert clicks == []


def test_dry_run_never_clicks(tracker):
    t, clicks = _tapper(tracker, dry_run=True)
    r = t.tap(0.5, 0.5)
    assert not r.sent and clicks == []


def test_failsafe_blocks_click(tracker):
    t, clicks = _tapper(tracker)

    def boom(margin):
        raise FailsafeTriggered("corner")

    t.failsafe = boom
    with pytest.raises(FailsafeTriggered):
        t.tap(0.5, 0.5)
    assert clicks == []
