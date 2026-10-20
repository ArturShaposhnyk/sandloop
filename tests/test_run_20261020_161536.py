"""Regressions from the live run debug/runs/20261020-161536 (L187, level planner on).

The plan worked for 6 taps and re-planned twice, then step 3 "tap r4c4 purple" waited for the
belt (k 3 > 2) from 43.9 s to 76 s: yellow 37 % and orange 0 % (its sand buried under purple)
could not fill, so no bucket left. The simulator filled buckets from any sand in the contact
layer (orange at 7 % of it); real buckets get ~nothing below ~10 %. The orange stall clock
also restarted every other frame (one or both orange buckets read). TAP #1 (right after
activating iPhone Mirroring) "had no visible effect", and at 76 s the Mirroring window was gone
for 10 s and the bot stopped.

tests/data/run-20261020-161536/: 00060-wait.png (the stalled board, 3/5: orange/cream 38,
yellow 36, orange 0) and belt.json (the belt read per frame from 40 s on, events.jsonl).
"""
import dataclasses
import json
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

from config import Config, load_config  # noqa: E402
from levelplan import Plan, PlanRunner, Step, plan_level  # noqa: E402
from sim import SimConfig, grid_signature, picture_from_frame  # noqa: E402
from state import BeltTracker, ConveyorBucket, LevelState  # noqa: E402
from strategy import choose  # noqa: E402
from tests.test_bot import Clock, Controls, Failsafe, Tapper, make_bot, run, shot, S056  # noqa: E402
from tests.test_state import analyzer  # noqa: E402,F401  (fixture)

DATA = Path(__file__).resolve().parent / "data" / "run-20261020-161536"


def frame():
    img = cv2.imread(str(DATA / "00060-wait.png"))
    assert img is not None
    return img


def stalled_state(an):  # noqa: F811
    from vision import read_gameplay

    gv = read_gameplay(frame(), an.palette, an.cfg.vision, an.reader, log=lambda s: None,
                       icons=an.icons, hints=an.hints)
    return gv, LevelState.from_view(gv, 5, an.cfg.strategy.unknown_fill_need)


# --- the belt: stall clock, real fill speed -------------------------------------------------

def test_belt_tracker_sees_the_orange_stall():
    """Orange read as one bucket (0 % or 38 %) or both: no restart - it never grew."""
    rows = json.loads((DATA / "belt.json").read_text())
    t = BeltTracker(min_growth=2, forget_s=3.0)
    for r in rows:
        out = t.update([ConveyorBucket(c, f) for c, f in r["conveyor"]], r["t"])
    assert max(r["stalled_old"]["orange"] for r in rows) < 3.0     # what the run had
    assert out["orange"] > 25.0 and out["yellow"] > 30.0


def test_sim_belt_speed_follows_the_measured_rate():
    pc = load_config().planner
    sc = SimConfig(5, pc.lap_s, pc.fill_rate, pc.fill_min_exposed, pc.fill_max, pc.tap_s)
    assert sc.rate(0.07) == 0.0                     # L187 orange at 7 %: nothing in 30 s
    assert 10 < sc.rate(0.25) <= 30 and sc.rate(0.9) == pc.fill_max


def test_sim_picture_matches_the_vision_contact_layer(analyzer):  # noqa: F811
    """The plan's picture is cut at the real bottom edge, block = most frequent colour: its
    contact shares match vision (before: orange 26 % vs 7 %, the plan expected it to fill)."""
    _, st = stalled_state(analyzer)
    pic = picture_from_frame(frame(), analyzer.palette, analyzer.cfg.vision,
                             analyzer.cfg.planner.cell_px)
    for colour, share in st.exposed.items():
        assert abs(pic.share(colour) - share) < 0.05, colour
    assert pic.share("orange") < analyzer.cfg.planner.fill_min_exposed + 0.02


def test_plan_from_the_stalled_board_taps_instead_of_waiting(analyzer):  # noqa: F811
    _, st = stalled_state(analyzer)
    pic = picture_from_frame(frame(), analyzer.palette, analyzer.cfg.vision,
                             analyzer.cfg.planner.cell_px)
    plan = plan_level(st, pic, analyzer.cfg.planner)
    first = plan.steps[0]
    assert first.belt_before == st.conveyor_k == 3 and first.t == 0.0    # a tap right now
    assert first.colour not in ("orange", "yellow")
    assert all(s.belt_before < 5 for s in plan.steps)
    yellow = [f for c, f in first.belt if c == "yellow"]
    assert yellow == [36] or yellow == [35]          # the plan knows yellow does not move


# --- following the plan: a wait never stalls ------------------------------------------------

def waiting_runner(cfg, st, belt_before=2, t=2.0):
    """A runner whose plan waits on this board for the belt to drop to ``belt_before``."""
    runner = PlanRunner(cfg)
    runner.signature = "test"
    pos = next(p for p in st.tappable if st.cell(p).color == "purple")
    runner.plan = Plan([Step(pos, "purple", belt_before, grid_signature(st.grid), t,
                             [("orange", 60), ("yellow", 80)])], "horizon", 0.0, 0.0, 0)
    runner.last_plan_t = 0.0
    runner.log = [{"t": 0.0, "why": "start"}]
    return runner


def test_plan_wait_gives_up_when_no_bucket_gains_sand(analyzer):  # noqa: F811
    gv, st = stalled_state(analyzer)
    cfg = analyzer.cfg
    runner, logs = waiting_runner(cfg, st), []
    fallback = choose(st, cfg.strategy)
    d = runner.decide(frame(), gv, st, fallback, 1.0, logs.append)
    assert d.move is None and "waits for the belt" in d.reason
    st.belt_stalled = {"yellow": 30.0, "orange": cfg.planner.wait_stall_s + 1}
    d = runner.decide(frame(), gv, st, fallback, 1.0 + cfg.planner.wait_stall_s + 1, logs.append)
    assert any("gives up waiting: no belt bucket gained sand" in s for s in logs)
    assert runner.log[0]["checks"][0]["gave_up"].startswith("no belt bucket gained sand")
    assert runner.log[-1]["why"] == "wait stalled"
    assert d.move is not None and d.move.level_plan == f"1/{len(runner.plan.steps)}"
    assert d.move.color not in ("orange", "yellow")


def test_plan_wait_gives_up_when_the_belt_is_behind_the_plan(analyzer):  # noqa: F811
    gv, st = stalled_state(analyzer)
    cfg = analyzer.cfg
    runner, logs = waiting_runner(cfg, st), []
    fallback = choose(st, cfg.strategy)
    runner.decide(frame(), gv, st, fallback, 1.0, logs.append)        # waiting since 1.0
    late = 1.0 + runner.predicted_wait(0) + cfg.planner.wait_grace_s
    d = runner.decide(frame(), gv, st, fallback, late - 0.5, logs.append)
    assert d.move is None and not any("gives up" in s for s in logs)
    runner.decide(frame(), gv, st, fallback, late + 0.5, logs.append)
    assert any("gives up waiting: the belt is behind the plan" in s for s in logs)


def test_when_the_new_plan_waits_too_the_strategy_decides(analyzer, monkeypatch):  # noqa: F811
    gv, st = stalled_state(analyzer)
    cfg = analyzer.cfg
    runner, logs = waiting_runner(cfg, st), []
    same = runner.plan
    monkeypatch.setattr("levelplan.plan_level", lambda *a, **k: same)
    fallback = choose(st, cfg.strategy)
    runner.decide(frame(), gv, st, fallback, 1.0, logs.append)
    d = runner.decide(frame(), gv, st, fallback, 30.0, logs.append)
    assert d is fallback and any("the strategy decides until the board changes" in s
                                 for s in logs)
    n = len(logs)
    assert runner.decide(frame(), gv, st, fallback, 31.0, logs.append) is fallback
    assert len(logs) == n                                    # no re-plan on the same board


def test_every_planned_tap_logs_predicted_vs_real_belt(analyzer):  # noqa: F811
    gv, st = stalled_state(analyzer)
    runner, logs = waiting_runner(analyzer.cfg, st, belt_before=3, t=4.0), []
    d = runner.decide(frame(), gv, st, choose(st, analyzer.cfg.strategy), 6.0, logs.append)
    assert d.move is not None
    (check,) = runner.log[0]["checks"]
    assert check["t"] == 6.0 and check["t_plan"] == 4.0 and check["k"] == 3
    assert check["belt_plan"] == [["orange", 60], ["yellow", 80]]
    assert any(s.startswith("plan check step 1/1: t 6s (plan 4s)") for s in logs)
    json.dumps(runner.log)


# --- focus and window ---------------------------------------------------------------------

class _Win:
    pid = 42


class _Tracker:
    window = _Win()

    def get(self, force=False):
        return None


def test_tapper_activates_mirroring_before_the_first_tap(monkeypatch):
    import actions

    front, calls = [0], []
    monkeypatch.setattr(actions, "activate_app", lambda *a: calls.append(a) or front.__setitem__(0, 42) or True)
    monkeypatch.setattr(actions, "frontmost_pid", lambda: front[0])
    cfg = Config()
    cfg.clicks = dataclasses.replace(cfg.clicks, focus_wait_s=0.2, focus_settle_s=0.0)
    logs = []
    t = actions.Tapper(cfg, _Tracker(), clicker=object(), log=logs.append)
    t.prepare()
    assert len(calls) == 1 and logs == ["iPhone Mirroring activated before the first tap"]
    t.prepare()                                              # frontmost already: no activation
    assert len(calls) == 1 and logs[-1] == "iPhone Mirroring is frontmost"
    actions.Tapper(cfg, _Tracker(), dry_run=True, log=logs.append).prepare()
    assert len(calls) == 1


def test_bot_waits_for_the_window_to_come_back(tmp_path):
    from bot import Bot
    from capture import WindowNotFound
    from runlog import RunLog
    from tests.test_bot import Geo

    cfg = load_config()
    cfg.planner = dataclasses.replace(cfg.planner, enabled=False)
    from analyzer import Analyzer

    an = Analyzer(cfg, clock=lambda: 0.0)
    clk, controls, g = Clock(), Controls(), shot(S056)
    calls = [0]

    def grab():
        calls[0] += 1
        if calls[0] > 42:
            controls.quit.set()
        if 2 < calls[0] < 40:                     # ~20 s without the window (0.5 s per try)
            raise WindowNotFound("no on-screen window owned by 'iPhone Mirroring'")
        return g

    log = RunLog(tmp_path / "debug", cfg, {"dry_run": True}, clock=clk, stamp="test")
    bot = Bot(cfg, an, Tapper(), grab, lambda force=False: Geo((g.shape[1], g.shape[0])), log,
              controls, Failsafe(), dry_run=True, clock=clk, sleep=clk.sleep,
              diagnose=lambda: "1 window(s) of 'iPhone Mirroring': #1 off-screen")
    result = run(bot, log)
    text = (log.dir / "bot.log").read_text()
    assert "window list: 1 window(s) of 'iPhone Mirroring': #1 off-screen" in text
    assert "window back after" in text and "window lost for" not in text
    assert list(log.dir.glob("frames/*-before-window-lost.png"))
    assert result == "stopped"                          # by the quit at the end, not the window


def test_first_no_effect_frame_is_checked_once_more(analyzer, tmp_path):  # noqa: F811
    g = shot(S056)
    cfg = load_config()
    cfg.bot = dataclasses.replace(cfg.bot, settle_min_s=0.0, settle_max_s=0.0,
                                  retap_guard_s=0.0, loop_interval_s=0.05)
    cfg.planner = dataclasses.replace(cfg.planner, enabled=False)
    tapper = Tapper()
    bot, log = make_bot(cfg, analyzer, tmp_path, [g] * 4, tapper=tapper)
    run(bot, log)
    text = (log.dir / "bot.log").read_text()
    assert "tap effect not visible yet - checking the next frame" in text
    assert text.index("not visible yet") < text.index("had no visible effect")
