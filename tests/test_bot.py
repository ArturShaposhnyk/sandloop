"""bot.py loop, tempo.py and tools/analyze_run.py on saved screenshots (no live game)."""
import dataclasses
import json
import threading
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

from actions import OutOfBounds, TapResult
from config import ConfigError, from_dict, load_config
from runlog import RunLog, load_events
from safety import FailsafeTriggered
from state import ConveyorBucket
from tempo import ExposureTrend, ReleaseLedger, burst, prioritize

SHOTS = Path(__file__).resolve().parent.parent / "dataset" / "screens"
S056 = "20261016-185258-702-056-live.png"    # L183, 1/5: only r1c1 cyan tappable
S052 = "20261016-113331-000-052-live.png"    # L181, 1/5: darkred at the bottom
S039 = "20261016-102536-802-039-auto.png"    # LEVEL FAILED
S023 = "20261016-102418-506-023-auto.png"    # home map


def shot(name):
    img = cv2.imread(str(SHOTS / name))
    assert img is not None, name
    return img


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class Controls:
    def __init__(self):
        self.paused, self.quit = threading.Event(), threading.Event()


class Failsafe:
    def __init__(self, trip_at=None):
        self.calls, self.trip_at = 0, trip_at

    def check(self):
        self.calls += 1
        if self.trip_at is not None and self.calls >= self.trip_at:
            raise FailsafeTriggered("cursor in a corner")


class Tapper:
    def __init__(self, refuse=False):
        self.taps, self.refuse = [], refuse

    def tap(self, rx, ry):
        if self.refuse:
            raise OutOfBounds("outside")
        self.taps.append((rx, ry))
        return TapResult((rx, ry), (rx * 300, ry * 650), True)


class Geo:
    def __init__(self, size):
        self.frame_size = size


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c.bot = dataclasses.replace(c.bot, settle_min_s=0.0, settle_max_s=0.0, retap_guard_s=0.0,
                                loop_interval_s=0.05)
    return c


@pytest.fixture(scope="module")
def analyzer(cfg):
    from analyzer import Analyzer

    return Analyzer(cfg, clock=lambda: 0.0)   # direct analyze() calls: no wall-clock stalls


def make_bot(cfg, analyzer, tmp_path, frames, tapper=None, failsafe=None, controls=None,
             geo_size=None, max_taps=None, dry_run=False, levels=1):
    from bot import Bot

    clk = Clock()
    it = iter(frames)
    last = [frames[0]]

    def grab():
        try:
            last[0] = next(it)
        except StopIteration:
            controls_.quit.set()
        return last[0]

    controls_ = controls or Controls()
    log = RunLog(tmp_path / "debug", cfg, {"dry_run": dry_run}, clock=clk, stamp="test")
    def geometry(force=False):   # the window as it is now: the last grabbed frame's size
        return Geo(geo_size or (last[0].shape[1], last[0].shape[0]))

    bot = Bot(cfg, analyzer, tapper or Tapper(), grab, geometry, log,
              controls_, failsafe or Failsafe(), dry_run=dry_run, max_taps=max_taps,
              levels=levels, clock=clk, sleep=clk.sleep)
    return bot, log


def run(bot, log):
    try:
        return bot.run()
    finally:
        log.close()


def test_confirmed_move_is_tapped_then_fail_screen_stops(cfg, analyzer, tmp_path):
    g, f = shot(S056), shot(S039)
    tapper = Tapper()
    bot, log = make_bot(cfg, analyzer, tmp_path, [g, g, f, f, f], tapper=tapper)
    assert run(bot, log) == "fail"
    assert len(tapper.taps) == 1                       # 2-frame confirmation, then one tap
    events = load_events(log.dir)
    tap = next(e for e in events if e["type"] == "tap")
    assert tap["pos"] == [1, 1] and tap["color"] == "cyan" and tap["k_before"] == 1
    assert (log.dir / tap["frame_png"]).exists() and (log.dir / tap["annotated_png"]).exists()
    res = json.loads((log.dir / "result.json").read_text())
    assert res["result"] == "fail" and res["taps"] == 1
    assert (log.dir / "belt.csv").read_text().count("\n") >= 3


def test_no_effect_taps_stop_the_bot(cfg, analyzer, tmp_path):
    g = shot(S056)
    tapper = Tapper()
    bot, log = make_bot(cfg, analyzer, tmp_path, [g] * 8, tapper=tapper)
    assert run(bot, log) == "stopped"
    res = json.loads((log.dir / "result.json").read_text())
    assert "no effect" in res["reason"] and len(tapper.taps) == cfg.bot.max_no_effect_taps


def test_dry_run_never_stops_for_missing_effect_and_retap_guard(cfg, analyzer, tmp_path):
    g = shot(S056)
    c = dataclasses.replace(cfg, bot=dataclasses.replace(cfg.bot, retap_guard_s=60.0))
    tapper = Tapper()
    bot, log = make_bot(c, analyzer, tmp_path, [g] * 6, tapper=tapper, dry_run=True)
    assert run(bot, log) == "stopped"                   # frames ran out -> quit
    assert len(tapper.taps) == 1                        # same cell guarded afterwards


def test_burst_on_l180_start(cfg, analyzer, tmp_path):
    g = shot("20261016-102033-862-001-auto.png")       # 0/5, blue at the bottom
    tapper = Tapper()
    bot, log = make_bot(cfg, analyzer, tmp_path, [g, g], tapper=tapper, max_taps=3)
    run(bot, log)
    taps = [e for e in load_events(log.dir) if e["type"] == "tap"]
    assert [t["burst"] for t in taps] == [1, 2, 3]
    assert len({tuple(t["pos"]) for t in taps}) == 3
    assert [t["free_before"] for t in taps] == [5, 4, 3]


def test_geometry_change_blocks_the_tap(cfg, analyzer, tmp_path):
    g = shot(S056)
    tapper = Tapper()
    bot, log = make_bot(cfg, analyzer, tmp_path, [g, g], tapper=tapper, geo_size=(500, 1000))
    run(bot, log)
    assert tapper.taps == []


def test_failsafe_and_quit_stop_without_taps(cfg, analyzer, tmp_path):
    g = shot(S056)
    tapper = Tapper()
    bot, log = make_bot(cfg, analyzer, tmp_path / "a", [g, g], tapper=tapper,
                        failsafe=Failsafe(trip_at=1))
    assert run(bot, log) == "stopped" and tapper.taps == []
    controls = Controls()
    controls.quit.set()
    bot, log = make_bot(cfg, analyzer, tmp_path / "b", [g, g], tapper=tapper, controls=controls)
    assert run(bot, log) == "stopped" and tapper.taps == []


def test_out_of_bounds_tap_is_refused(cfg, analyzer, tmp_path):
    g = shot(S056)
    bot, log = make_bot(cfg, analyzer, tmp_path, [g, g], tapper=Tapper(refuse=True))
    run(bot, log)
    assert bot.taps == 0


def test_menu_before_gameplay_taps_play(cfg, analyzer, tmp_path):
    m = shot(S023)
    tapper = Tapper()
    bot, log = make_bot(cfg, analyzer, tmp_path, [m] * 3, tapper=tapper)
    run(bot, log)
    assert len(tapper.taps) == 1 and 0.4 < tapper.taps[0][0] < 0.6   # PLAY
    tap = next(e for e in load_events(log.dir) if e["type"] == "screen_tap")
    assert tap["screen"] == "menu" and tap["button"] == "play" and tap["lives"] == 5
    assert (log.dir / tap["frame_png"]).exists() and (log.dir / tap["annotated_png"]).exists()


# --- phase 6: level after level ---------------------------------------------------------------

S059 = "20261019-171450-497-059-auto.png"    # Well Done! NEXT / Watch Ad
S036 = "20261016-102521-772-036-auto.png"    # revive: Quit / Keep Playing 900
S001 = "20261016-102033-862-001-auto.png"    # L180 start


@pytest.fixture(scope="module")
def cfg6(cfg):
    return dataclasses.replace(cfg, flow=dataclasses.replace(cfg.flow, tap_cooldown_s=0.0))


def screen_taps(log):
    return [(e["screen"], e["button"]) for e in load_events(log.dir) if e["type"] == "screen_tap"]


def test_one_level_stops_on_the_win_screen_without_tapping(cfg6, analyzer, tmp_path):
    g, w = shot(S056), shot(S059)
    tapper = Tapper()
    bot, log = make_bot(cfg6, analyzer, tmp_path, [g, g] + [w] * 4, tapper=tapper)
    assert run(bot, log) == "win"
    assert len(tapper.taps) == 1 and screen_taps(log) == []
    res = json.loads((log.dir / "result.json").read_text())
    assert res["levels"] == 1 and res["wins"] == 1


def test_win_next_play_next_level(cfg6, analyzer, tmp_path):
    g, w, m, g2 = shot(S056), shot(S059), shot(S023), shot(S001)
    tapper = Tapper()
    frames = [g, g] + [w] * 6 + [m] * 3 + [g2]
    bot, log = make_bot(cfg6, analyzer, tmp_path, frames, tapper=tapper, levels=2)
    run(bot, log)
    assert screen_taps(log) == [("level_complete", "next"), ("menu", "play")]
    from tools.analyze_run import post_mortem

    text = "\n".join(post_mortem(log.dir))
    assert "levels: L1 win" in text and "level_complete: next" in text
    events = load_events(log.dir)
    assert [e["level"] for e in events if e["type"] == "level_start"] == [1, 2]
    assert [e["result"] for e in events if e["type"] == "level_end"] == ["win"]
    nx = next(e for e in events if e["type"] == "screen_tap")
    assert nx["rel"][0] > 0.6                            # NEXT, not Watch Ad (left)
    # phase 7: every level -> debug/levels.jsonl (L2 is cut off by the end of the frames)
    lv = [json.loads(l) for l in (log.root / "levels.jsonl").read_text().splitlines()]
    assert [(r["level"], r["result"]) for r in lv] == [(1, "win"), (2, "unfinished")]
    first = lv[0]
    assert first["run"] == log.dir.name and first["taps"] == 1 and first["strategy"]
    tap = first["tap_log"][0]
    assert tap["type"] == "tap" and tap["decision"]["candidates"]
    assert first["end"]["grid"] and "lookahead" in tap["decision"]["candidates"][0]


def test_revive_quit_then_try_again(cfg6, analyzer, tmp_path):
    g, r, f = shot(S056), shot(S036), shot(S039)
    tapper = Tapper()
    frames = [g, g] + [r] * 5 + [f] * 4
    bot, log = make_bot(cfg6, analyzer, tmp_path, frames, tapper=tapper, levels=3)
    run(bot, log)
    assert screen_taps(log) == [("revive", "quit"), ("fail", "try_again")]
    ends = [e for e in load_events(log.dir) if e["type"] == "level_end"]
    assert [e["result"] for e in ends] == ["fail"]       # revive + fail = one lost level


def test_consecutive_fails_stop_the_session(cfg6, analyzer, tmp_path):
    c = dataclasses.replace(cfg6, session=dataclasses.replace(cfg6.session,
                                                              max_consecutive_fails=1))
    g, f = shot(S056), shot(S039)
    tapper = Tapper()
    bot, log = make_bot(c, analyzer, tmp_path, [g, g] + [f] * 4, tapper=tapper, levels=5)
    assert run(bot, log) == "fail"
    assert screen_taps(log) == []
    assert "lost in a row" in json.loads((log.dir / "result.json").read_text())["reason"]


def test_unknown_screen_too_long_stops_with_frame(cfg6, analyzer, tmp_path):
    c = dataclasses.replace(cfg6, flow=dataclasses.replace(cfg6.flow, unknown_max_s=0.3))
    splash = shot("20261016-102416-612-022-auto.png")
    tapper = Tapper()
    bot, log = make_bot(c, analyzer, tmp_path, [splash] * 12, tapper=tapper)
    assert run(bot, log) == "stopped"
    res = json.loads((log.dir / "result.json").read_text())
    assert "screen other for" in res["reason"] and tapper.taps == []
    assert any(p.name.endswith("stop-other.png") for p in (log.dir / "frames").iterdir())


def test_analyze_run_post_mortem(cfg, analyzer, tmp_path):
    from tools.analyze_run import latest_run, post_mortem

    g, f = shot(S056), shot(S039)
    bot, log = make_bot(cfg, analyzer, tmp_path, [g, g, f, f, f])
    run(bot, log)
    assert latest_run(tmp_path / "debug") == log.dir
    text = "\n".join(post_mortem(log.dir))
    assert "result: FAIL" in text and "r1c1 cyan" in text and "belt at the end" in text


# --- tempo ------------------------------------------------------------------------------------

def test_exposure_trend_covering():
    tr = ExposureTrend(window_s=2.0, min_drop=0.03)
    tr.update(0.0, {"red": 0.30, "blue": 0.20})
    tr.update(1.0, {"red": 0.20, "blue": 0.19})
    assert tr.covering() == {"red": pytest.approx(0.10)}
    assert tr.covering(on_belt=["red"]) == {}           # our own bucket collecting it
    tr.update(5.0, {"red": 0.20})
    assert tr.covering() == {}                          # old samples dropped


def test_prioritize_prefers_covered_colour(cfg, analyzer):
    res = analyzer.analyze(shot(S052), log=lambda s: None)
    allowed = [c for c in res.decision.candidates if c.rejected is None]
    other = next(c for c in allowed if c.color != res.decision.move.color)
    d = prioritize(res.decision, {other.color: 1.0}, 1.0)
    assert d.move.color == other.color and "being covered" in d.reason


def test_burst_never_takes_the_last_slot(cfg, analyzer):
    res = analyzer.analyze(shot(S052), log=lambda s: None)   # 1/5
    moves = burst(res.state, res.decision.move, set(res.state.tappable), cfg.strategy, cfg.bot)
    assert 1 < len(moves) <= cfg.bot.burst_max
    assert res.state.free_slots - (len(moves) - 1) >= cfg.bot.burst_min_free


def test_ledger_tower_stages_and_guard():
    from state import LevelState

    led = ReleaseLedger(retap_guard_s=2.0)
    led.add(0.0, (1, 2), "cream", "red")
    assert led.guarded((1, 2), 1.0) and not led.guarded((1, 2), 2.5)
    st = LevelState([], [], [], [], [], 1, 5, [ConveyorBucket("cream", 40)])
    led.update(st, 1.0)
    assert st.conveyor[0].stages == ("red",)
    st = LevelState([], [], [], [], [], 1, 5, [ConveyorBucket("red", 10)])
    led.update(st, 2.0)                                  # cream done, red collecting
    assert led.released[0].color == "red" and st.conveyor[0].stages == ()


def test_bot_config_validation():
    with pytest.raises(ConfigError):
        from_dict({"bot": {"burst_min_free": 1}})
    with pytest.raises(ConfigError):
        from_dict({"bot": {"settle_zones": ["sky"]}})
    with pytest.raises(ConfigError):
        from_dict({"bot": {"settle_min_s": 2.0, "settle_max_s": 1.0}})
    assert from_dict({"bot": {"burst_max": 1}}).bot.burst_max == 1
