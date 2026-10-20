"""Whole-level planner (sim.py, levelplan.py) and the run-20261020-145310 strategy fixes: per-colour
capacity, the endgame slot reserve, burst follow-ups, the sacrifice rule. Run 145310 (L187): the
cream last-slot gamble opened purple and the sky drained, but the level was lost with the belt
full of almost-done colours while two darkred buckets (39 % of the contact layer) had no slot."""
import dataclasses
import json
import shutil
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

from analyzer import colour_capacity  # noqa: E402
from config import StrategyConfig, load_config  # noqa: E402
from levelplan import LevelMemory, PlanRunner, _record_signature, plan_level  # noqa: E402
from sim import Picture, Sim, SimConfig, grid_signature, picture_from_frame  # noqa: E402
from state import ConveyorBucket, LevelState  # noqa: E402
from strategy import choose  # noqa: E402
from tempo import burst  # noqa: E402
from tests.test_state import analyzer, make_state  # noqa: E402,F401  (fixture)

DATA = Path(__file__).resolve().parent / "data" / "run-20261020-145310"
# the belt stall clocks the run had at these frames (events.jsonl)
STALLED = {"00134-tap": {"blue": 13.0, "orange": 0.0, "yellow": 0.0},
           "00188-tap": {"blue": 10.0, "darkgreen": 28.0, "purple": 2.0, "yellow": 2.0},
           "00201-tap": {"darkgreen": 39.0, "orange": 1.0, "yellow": 2.0, "blue": 6.0}}


def frame(name):
    img = cv2.imread(str(DATA / f"{name}.png"))
    assert img is not None, name
    return img


def read(an, name):  # noqa: F811
    from vision import read_gameplay

    return read_gameplay(frame(name), an.palette, an.cfg.vision, an.reader, log=lambda s: None,
                         icons=an.icons, hints=an.hints)


def run_state(an, name):  # noqa: F811
    """A frame of the run with the capacity per colour measured at its start (as live)."""
    start = LevelState.from_view(read(an, "00002-tap"), 5, an.cfg.strategy.unknown_fill_need)
    st = LevelState.from_view(read(an, name), 5, an.cfg.strategy.unknown_fill_need,
                              STALLED.get(name))
    st.colour_capacity = colour_capacity(start)
    return st


# --- per-colour capacity / endgame / burst / sacrifice ------------------------------------------

def test_capacity_per_colour_cancels_the_reading_bias(analyzer):  # noqa: F811
    """darkred reads 3.6 % for 3 stages (1.2 %/bucket) while orange reads 3.9 %/bucket."""
    start = LevelState.from_view(read(analyzer, "00002-tap"))
    cap = colour_capacity(start)
    assert cap["darkred"] < 0.5 * cap["orange"]
    st = run_state(analyzer, "00201-tap")
    darkred = next(c for c in choose(st, analyzer.cfg.strategy).candidates if c.color == "darkred")
    assert "not enough sand" not in (darkred.rejected or "")   # the run rejected it to the end


def test_endgame_keeps_the_last_slot_for_darkred(analyzer):  # noqa: F811
    """#26: purple (27 % exposed, almost done) took the last slot; darkred is 39 %."""
    d = choose(run_state(analyzer, "00201-tap"), analyzer.cfg.strategy)
    assert d.move is not None and d.move.color == "darkred"
    purple = next(c for c in d.candidates if c.color == "purple")
    assert purple.rejected.startswith("endgame: keep a slot for darkred")


def test_endgame_reserve_never_makes_two_colours_wait_for_each_other():
    sc = StrategyConfig()
    st = make_state(["red blue"], k=4, conveyor=[ConveyorBucket("green", 50)] * 4,
                    sand={"red": .01, "blue": .01, "green": .5},
                    exposed={"red": .4, "blue": .3, "green": .2})
    st.colour_capacity = {"red": .02, "blue": .02, "green": .2}
    d = choose(st, sc)
    assert d.move is not None and d.move.color == "red"


def test_burst_follow_up_needs_a_good_score(analyzer):  # noqa: F811
    """#24 orange + burst #25 yellow (0.15, sand nearly gone): no such follow-up any more."""
    st = run_state(analyzer, "00188-tap")
    d = choose(st, analyzer.cfg.strategy)
    moves = burst(st, d.move, set(st.tappable), analyzer.cfg.strategy, analyzer.cfg.bot)
    assert all(m.color != "yellow" for m, _ in moves)
    assert all(m.score >= analyzer.cfg.bot.burst_min_score for m, why in moves[1:])


def test_sacrifice_a_slot_to_open_what_unblocks_the_belt():
    """User (a): cream cannot fill (sand short) but opens purple, whose collection uncovers the
    yellow stalled on the belt; a slot stays free after it."""
    sc = StrategyConfig()
    st = make_state(["cream", "purple_"], k=1, conveyor=[ConveyorBucket("yellow", 40)],
                    sand={"purple": .3, "cream": .001, "yellow": .2},
                    exposed={"cream": .2, "purple": .3})
    st.belt_stalled = {"yellow": 10.0}
    st.above = {"purple": {"yellow": .2}}
    d = choose(st, sc)
    assert d.move is not None and d.move.sacrifice and d.move.color == "cream"
    st.conveyor_k = 4                                         # only the last slot: no sacrifice
    st.conveyor = [ConveyorBucket("yellow", 40)] * 4
    d = choose(st, sc)
    assert d.move is None or not d.move.sacrifice


# --- simulator --------------------------------------------------------------------------------

def test_picture_take_lets_the_sand_above_fall():
    pic = Picture(["a", "b"], [[0, 1, 0], [1, 1]], contact=1)
    assert pic.exposed("a") == 1 and pic.take("a", 5) == 1
    assert pic.cols == [[1, 0], [1, 1]] and pic.exposed("b") == 2


def test_sim_jams_when_the_belt_is_full_of_buried_colours():
    st = make_state(["a b c d e"], k=0, sand={"a": .1})
    pic = Picture(["a", "b", "c", "d", "e"], [[0, 1, 2, 3, 4]] * 4, contact=1)
    s = Sim.start(st, pic, SimConfig(slots=5, lap_take=10))
    for p in [(0, 1), (0, 2), (0, 3), (0, 4)]:
        s = s.tap(p)
    s = s.tap((0, 0))                                         # a collects, the rest is buried
    assert len(s.belt) < 5
    full = Sim.start(make_state(["b c d e x"]), Picture(["a", "b", "c", "d", "e"],
                                                        [[0, 0, 1, 2, 3, 4]] * 4, 1),
                     SimConfig(slots=4, lap_take=10))
    for p in [(0, 0), (0, 1), (0, 2), (0, 3)]:
        full = full.tap(p)
    w = full.wait()
    assert w is not None and w.jam


# --- planner ----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def start_plan(analyzer):  # noqa: F811
    st = LevelState.from_view(read(analyzer, "00002-tap"))
    pic = picture_from_frame(frame("00002-tap"), analyzer.palette, analyzer.cfg.vision,
                             analyzer.cfg.planner.cell_px)
    return st, pic, plan_level(st, pic, analyzer.cfg.planner)


def test_plan_from_the_l187_start_never_fills_the_belt(start_plan):
    st, _, plan = start_plan
    assert len(plan.steps) >= 10 and plan.result in ("win", "horizon")
    assert plan.dead == 0
    assert all(s.belt_before < 5 for s in plan.steps)
    assert plan.steps[0].pos in st.tappable
    assert len({s.pos for s in plan.steps}) == len(plan.steps)


def test_plan_avoids_a_banned_sequence(start_plan, analyzer):  # noqa: F811
    st, pic, plan = start_plan
    seq = tuple(s.pos for s in plan.steps)
    again = plan_level(st, pic, analyzer.cfg.planner, banned={seq})
    assert tuple(s.pos for s in again.steps) != seq


def test_level_memory_reads_the_run(tmp_path):
    """The run's own levels.jsonl line (written before records had a signature)."""
    mem = LevelMemory(DATA / "levels.jsonl")
    (rec,) = mem.records
    sig = _record_signature(rec)
    assert rec["result"] == "fail" and len(next(iter(mem.banned(sig)))) == 26
    caps = mem.capacity_shares(sig)
    assert 0.02 < caps["darkred"] < 0.035 and 0.03 < caps["orange"] < 0.05
    assert LevelMemory(tmp_path / "none.jsonl").banned(sig) == set()


def test_plan_runner_follows_and_replans(analyzer, tmp_path):  # noqa: F811
    cfg = dataclasses.replace(analyzer.cfg)
    shutil.copy(DATA / "levels.jsonl", tmp_path / "levels.jsonl")
    runner = PlanRunner(cfg, tmp_path / "levels.jsonl")
    gv = read(analyzer, "00002-tap")
    st = LevelState.from_view(gv)
    logs = []
    fallback = choose(st, cfg.strategy)
    d = runner.decide(frame("00002-tap"), gv, st, fallback, 0.0, logs.append)
    assert any("1 failed sequence(s) avoided" in s for s in logs)
    first = runner.plan.steps[0]
    assert d.move is not None and d.move.pos == first.pos and d.move.level_plan == f"1/{len(runner.plan.steps)}"
    after = st.after_tap(first.pos)                           # the board of step 2
    after.conveyor_k = runner.plan.steps[1].belt_before
    d2 = runner.decide(frame("00002-tap"), gv, after, fallback, 1.0, logs.append)
    assert d2.reason.endswith(f"(level plan step 2/{len(runner.plan.steps)}, {runner.plan.result})") \
        or "waits for the belt" in d2.reason or "not tappable" in d2.reason
    other = st.after_tap(st.tappable[-1]).after_tap(st.tappable[0])   # a board off the plan
    other.conveyor_k = 0
    if grid_signature(other.grid) not in {s.board for s in runner.plan.steps}:
        runner.decide(frame("00002-tap"), gv, other, fallback, 5.0, logs.append)
        assert runner.replans == 1 and any("board differs" in s for s in logs)
    rec = runner.log[0]
    assert rec["why"] == "start" and rec["banned"] == 1 and rec["steps"]
    json.dumps(runner.log)                                   # goes to levels.jsonl


def test_config_has_the_planner():
    cfg = load_config()
    assert cfg.planner.enabled and cfg.planner.beam >= 1
