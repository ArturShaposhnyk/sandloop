"""Phase 6: screens outside gameplay (flow.py, adclose.py, hud.py) on saved screenshots."""
import dataclasses
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

from adclose import find_close
from config import SessionConfig, load_config
from digits import DigitReader
from flow import AD_CLOSE, ScreenFlow, Session, plan
from hud import read_lives
from palette import Palette
from screens import FORBIDDEN, Match, ScreenClassifier, ScreenResult, load_templates

ROOT = Path(__file__).resolve().parents[1]
SCREENS = ROOT / "dataset" / "screens"


def shot(seq):
    hits = [p for p in SCREENS.glob(f"*-{seq}-*.png") if p.stem.split("-")[3] == seq]
    assert len(hits) == 1, seq
    return cv2.imread(str(hits[0]))


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "config.yaml")


@pytest.fixture(scope="module")
def clf(cfg):
    return ScreenClassifier(load_templates(ROOT / "dataset" / "templates"), cfg.vision)


@pytest.fixture(scope="module")
def hud(cfg):
    return (Palette.load(ROOT / cfg.paths.palette),
            DigitReader.load(ROOT / cfg.paths.dataset / "digits.npz"))


# (seq, expected action kind, button) with 5 lives read
EXPECTED = [
    ("059", "tap", "next"), ("060", "tap", "next"), ("061", "tap", "next"),
    ("062", "tap", "next"), ("063", "tap", "next"),
    ("023", "tap", "play"), ("028", "tap", "play"),
    ("024", "tap", "close_red"), ("025", "tap", "close_red"), ("026", "tap", "close_red"),
    ("051", "tap", "close_red"),
    ("036", "tap", "quit"), ("037", "wait", ""), ("038", "tap", "close_red"),
    ("039", "tap", "try_again"),
    ("012", "wait", ""), ("013", "wait", ""), ("019", "wait", ""), ("020", "wait", ""),
    ("018", "tap", AD_CLOSE),
    ("021", "wait", ""), ("022", "wait", ""),
]


@pytest.mark.parametrize("seq,kind,button", EXPECTED, ids=[e[0] for e in EXPECTED])
def test_plan_per_screen(seq, kind, button, cfg, clf):
    img = shot(seq)
    res = clf.classify(img)
    act = plan(res, img, 5, cfg.flow)
    assert (act.kind, act.button) == (kind, button), act.reason
    if act.kind == "tap":
        for name in FORBIDDEN:   # never on Watch Ad / Keep Playing / Get App / App Store
            m = res.find(name)
            assert m is None or not (m.box_rel[0] <= act.rel[0] <= m.box_rel[2]
                                     and m.box_rel[1] <= act.rel[1] <= m.box_rel[3])


def test_win_screen_taps_next_not_watch_ad(cfg, clf):
    img = shot("059")
    res = clf.classify(img)
    assert res.screen == "level_complete"
    x, y = plan(res, img, None, cfg.flow).rel
    assert 0.62 < x < 0.88 and 0.77 < y < 0.83          # NEXT, right half
    wx0, _, wx1, _ = res.find("watch_ad").box_rel
    assert wx1 < 0.5 and not wx0 <= x <= wx1


def test_status_bar_marks_ads(clf):
    assert clf.classify(shot("020")).screen == "ad"     # end card: no Get App / mute, status bar


def test_revive_never_keep_playing(cfg, clf):
    img = shot("036")
    act = plan(clf.classify(img), img, 5, cfg.flow)
    assert act.button == "quit" and act.rel[0] < 0.2


def test_lives_gate_play_and_try_again(cfg, clf):
    for seq in ("023", "039"):
        img = shot(seq)
        res = clf.classify(img)
        assert plan(res, img, 0, cfg.flow).kind == "stop"
        assert plan(res, img, None, cfg.flow).kind == "wait"
        no_req = dataclasses.replace(cfg.flow, require_lives=False)
        assert plan(res, img, None, no_req).kind == "tap"


def test_button_inside_forbidden_box_is_not_tapped(cfg):
    res = ScreenResult("level_complete", {
        "next": Match("next", 0.99, (0.3, 0.8), (0.2, 0.78, 0.4, 0.82)),
        "watch_ad": Match("watch_ad", 0.99, (0.25, 0.8), (0.05, 0.77, 0.45, 0.83))})
    act = plan(res, shot("059"), 5, cfg.flow)
    assert act.kind == "wait" and "forbidden" in act.reason


def test_other_screen_is_searched_for_a_cross_only_after_an_ad(cfg):
    img = shot("018")
    res = ScreenResult("other", {})
    assert plan(res, img, None, cfg.flow).kind == "wait"
    act = plan(res, img, None, cfg.flow, ad_recent=True)
    assert act.button == AD_CLOSE


# --- adclose / hud ---------------------------------------------------------------------------

def test_ad_close_finder(cfg):
    marks = find_close(shot("018"), cfg.flow)
    assert marks and abs(marks[0].center_rel[0] - 0.868) < 0.01
    assert abs(marks[0].center_rel[1] - 0.939) < 0.01
    for seq in ("013", "019"):                          # the mute icon's small x is no ✕
        assert find_close(shot(seq), cfg.flow) == []
    for seq in ("001", "030", "041", "057", "023", "036", "039", "059"):
        assert find_close(shot(seq), cfg.flow) == [], seq


@pytest.mark.parametrize("seq,lives", [("001", 5), ("023", 5), ("036", 5), ("039", 5),
                                       ("041", 4), ("057", 4), ("024", None), ("059", None),
                                       ("013", None)])
def test_read_lives(seq, lives, cfg, hud):
    pal, reader = hud
    assert read_lives(shot(seq), pal, reader, cfg.flow) == lives


# --- ScreenFlow ------------------------------------------------------------------------------

@pytest.fixture
def menu(clf):
    img = shot("023")
    return clf.classify(img), img


def test_flow_confirms_then_cools_down(cfg, menu):
    res, img = menu
    fl = ScreenFlow(dataclasses.replace(cfg.flow, confirm_frames=3, tap_cooldown_s=2.0))
    kinds = [fl.feed(res, img, 5, t).kind for t in (0.0, 0.1)]
    assert kinds == ["wait", "wait"]
    act = fl.feed(res, img, 5, 0.2)
    assert act.kind == "tap" and act.button == "play"
    fl.tapped(act, 0.2)
    assert [fl.feed(res, img, 5, t).kind for t in (0.3, 0.4, 0.5)] == ["wait"] * 3
    assert fl.feed(res, img, 5, 2.3).kind == "tap"      # still there after the cooldown


def test_flow_same_button_without_effect_stops(cfg, menu):
    res, img = menu
    fl = ScreenFlow(dataclasses.replace(cfg.flow, confirm_frames=1, tap_cooldown_s=0.0,
                                        same_button_max=2))
    for t in (0.0, 1.0):
        act = fl.feed(res, img, 5, t)
        assert act.kind == "tap"
        fl.tapped(act, t)
    act = fl.feed(res, img, 5, 2.0)
    assert act.kind == "stop" and "did not change" in act.reason
    fl.in_gameplay()                                     # a level in between resets it
    assert fl.feed(res, img, 5, 3.0).kind == "tap"


def test_flow_max_taps_between_levels(cfg, clf, menu):
    res, img = menu
    fl = ScreenFlow(dataclasses.replace(cfg.flow, confirm_frames=1, tap_cooldown_s=0.0,
                                        same_button_max=99, max_taps_between_levels=2))
    for t in (0.0, 1.0):
        fl.tapped(fl.feed(res, img, 5, t), t)
    assert fl.feed(res, img, 5, 2.0).kind == "stop"


def test_flow_timeouts(cfg, clf):
    fc = dataclasses.replace(cfg.flow, unknown_max_s=10.0, ad_max_s=30.0)
    fl = ScreenFlow(fc)
    other, img = ScreenResult("other", {}), shot("022")
    assert fl.feed(other, img, None, 0.0).kind == "wait"
    assert fl.feed(other, img, None, 9.0).kind == "wait"
    act = fl.feed(other, img, None, 10.5)
    assert act.kind == "stop" and "other" in act.reason
    fl = ScreenFlow(fc)
    ad = shot("013")
    res = clf.classify(ad)
    assert fl.feed(res, ad, None, 0.0).kind == "wait"
    assert fl.feed(res, ad, None, 25.0).kind == "wait"
    assert fl.feed(res, ad, None, 31.0).kind == "stop"


def test_flow_dry_run_logs_a_button_once(cfg, menu):
    res, img = menu
    fl = ScreenFlow(dataclasses.replace(cfg.flow, confirm_frames=1, tap_cooldown_s=0.0), True)
    act = fl.feed(res, img, 5, 0.0)
    assert act.kind == "tap"
    fl.tapped(act, 0.0)
    act = fl.feed(res, img, 5, 1.0)
    assert act.kind == "wait" and "dry run" in act.reason


def test_flow_lives_zero_stops_anywhere(cfg):
    fl = ScreenFlow(cfg.flow)
    assert fl.feed(ScreenResult("other", {}), shot("022"), 0, 0.0).kind == "stop"


# --- Session ---------------------------------------------------------------------------------

def test_session_levels_and_fails_in_a_row():
    s = Session(3, SessionConfig(max_levels=50, max_consecutive_fails=2))
    assert s.record("win") is None and s.record("fail") is None
    assert "3 level(s)" in s.record("win")
    s = Session(10, SessionConfig(max_levels=50, max_consecutive_fails=2))
    s.record("fail")
    assert "lost in a row" in s.record("fail")
    assert Session(99, SessionConfig(max_levels=5)).levels == 5
    assert s.summary()["fails"] == 2
