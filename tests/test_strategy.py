"""Greedy strategy: rules on synthetic states, decisions on saved shots, overlay dry-run."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from config import StrategyConfig
from cells import Kind
from state import ConveyorBucket
from strategy import choose
from tests.test_state import analyzer, make_state, shot  # noqa: F401  (fixture)

ROOT = Path(__file__).resolve().parents[1]
SC = StrategyConfig()


def by_pos(decision):
    return {c.pos: c for c in decision.candidates}


def test_waits_when_counter_unread_or_conveyor_full():
    st = make_state(["red"], k=None, sand={"red": .5}, exposed={"red": .5})
    assert choose(st, SC).move is None and "counter" in choose(st, SC).reason
    d = choose(make_state(["red"], k=5, sand={"red": .5}, exposed={"red": .5}), SC)
    assert d.move is None and d.reason == "wait: conveyor full"
    assert by_pos(d)[(0, 0)].rejected == "conveyor full"


def test_rejects_colour_whose_sand_is_not_exposed():
    # 2 free slots: the plain minimum applies (with 3+ free, or an idle belt, it is relaxed)
    st = make_state(["red blue"], k=3, sand={"red": .3, "blue": .3}, exposed={"blue": .5})
    d = choose(st, SC)
    assert d.move.pos == (0, 1)
    assert "not enough exposed sand" in by_pos(d)[(0, 0)].rejected


def test_last_free_slot_needs_strongly_exposed_sand():
    st = make_state(["red"], k=4, sand={"red": .5}, exposed={"red": .1})
    d = choose(st, SC)
    assert d.move is None and "last slot" in d.candidates[0].rejected
    st.exposed["red"] = .4
    assert choose(st, SC).move.pos == (0, 0)


def test_rejects_colour_without_enough_sand_left():
    # 4 buckets pending, total sand 0.4 -> capacity 0.1; a second red on the conveyor
    # needs (1 + 1) * 0.1 * 0.5 = 0.1 of red sand, only 0.02 is left
    st = make_state(["red blue blue"], k=1, conveyor=[ConveyorBucket("red", 0)],
                    sand={"red": .02, "blue": .38}, exposed={"red": .5, "blue": .2})
    d = choose(st, SC)
    assert "not enough sand" in by_pos(d)[(0, 0)].rejected
    assert d.move.color == "blue"


def test_prefers_tap_that_opens_buckets():
    st = make_state(["red red",
                     "x blue_"], k=0, sand={"red": .5, "blue": .5},
                    exposed={"red": .5, "blue": .5})
    d = choose(st, SC)
    assert d.move.pos == (0, 1) and d.move.opens == [(1, 1)]


def test_tie_goes_to_front_row_then_left():
    st = make_state(["red . red",
                     ". . ."], k=0, sand={"red": .5}, exposed={"red": .5})
    assert choose(st, SC).move.pos == (0, 0)


def test_only_tappable_cells_are_candidates():
    st = make_state(["red", "blue_"], k=0, sand={"red": .5, "blue": .5},
                    exposed={"red": .5, "blue": .5})
    st.tappable = []                       # e.g. visual state disagrees with the path rule
    d = choose(st, SC)
    assert d.move is None and d.candidates == [] and "no tappable" in d.reason


# --- on saved screenshots --------------------------------------------------------------------

@pytest.mark.parametrize("seq,want", [
    ("001", (0, 1)),     # L180 start: blue (only exposed colour), opens the pink behind it
    ("052", (2, 3)),     # live regression: red 49% on the belt starves, bottom is darkred
    ("008", (4, 0)),     # blue is the most exposed colour
    ("053", (0, 2)),     # L183 start: darkgreen (most of the grass at the bottom), user-confirmed
    ("054", (2, 5)),     # live regression: face-up darkgreen/cyan tower next to the yellow key
    ("055", (4, 0)),     # live: hint arrows; open LIGHT green read darkgreen from its inside (user)
    ("056", (1, 1)),     # live: 1/5, cyan sky just above the bottom; only move, opens a path (user)
])
def test_decision_on_shot(seq, want, analyzer):  # noqa: F811
    res = analyzer.analyze(shot(seq), log=lambda s: None)
    assert res.decision.move is not None and res.decision.move.pos == want
    x, y = res.tap_rel()
    assert 0 < x < 1 and 0.5 < y < 0.9


@pytest.mark.parametrize("seq", ["030", "041"])   # L181 start (first try and retry)
def test_tall_tower_top_is_face_up_and_chosen(seq, analyzer):  # noqa: F811
    """Live dry-run regression: the open cream top of the tall tower read face-down (its rim is
    drawn above the cell crop) and the bot waited with "no tappable bucket"."""
    res = analyzer.analyze(shot(seq), log=lambda s: None)
    cell = res.view.grid[0][2]
    assert (cell.color, cell.face_up) == ("cream", True)
    assert (0, 2) in res.view.tappable() and res.view.disagreements == []
    assert res.decision.move is not None and res.decision.move.pos == (0, 2)


@pytest.mark.parametrize("seq", ["006", "011", "033", "045"])   # k = 5/5
def test_waits_on_full_conveyor(seq, analyzer):  # noqa: F811
    assert analyzer.analyze(shot(seq), log=lambda s: None).decision.reason == "wait: conveyor full"


def test_never_taps_cream_when_its_sand_is_buried(analyzer):  # noqa: F811
    d = analyzer.analyze(shot("043"), log=lambda s: None).decision
    for pos in [(2, 1), (2, 4), (4, 3)]:
        assert "not enough exposed sand" in by_pos(d)[pos].rejected


def test_last_slot_not_spent_on_a_colour_the_belt_already_claims(analyzer):  # noqa: F811
    """043, 4/5: red is 31% of the contact layer but the red bucket on the belt (16%) still
    claims most of it, so a second red is not worth the last slot."""
    d = analyzer.analyze(shot("043"), log=lambda s: None).decision
    assert d.move is None and d.reason == "wait: no bucket passes the rules"
    red = by_pos(d)[(3, 2)]
    assert red.exposed > 0.2 and red.belt_demand > 0.15 and "last slot" in red.rejected


def test_live_regression_052_prefers_darkred_over_starving_red(analyzer):  # noqa: F811
    """L181 dry-run feedback: red 49% on the belt collects nothing (no red in the contact
    layer); the bottom is darkred. Expected: darkred (r2c3 or r4c1), red rejected."""
    res = analyzer.analyze(shot("052"), log=lambda s: None)
    d = res.decision
    assert d.move.color == "darkred" and d.move.pos in [(2, 3), (4, 1)]
    exp = res.state.exposed   # the reach band (lowest ~28 px of sand per column)
    assert max(exp, key=exp.get) == "darkred" and exp.get("red", 0) < 0.15
    red = by_pos(d)[(3, 2)]
    # the reach band sees some red a little higher up (12 %), but the red bucket on the belt
    # still claims more than that (13 %): a second red is rejected even with 4 free slots
    assert red.rejected and red.avail < 0
    assert "belt 13%" in red.describe()


def test_live_regression_054_face_up_tower_on_another_colour(analyzer):  # noqa: F811
    """L183 dry-run: an open darkgreen bucket on a cyan stage read face-down (its dark inside
    taken for a disc on the cyan dome) and the bot waited with "no tappable bucket". Tapping
    it sends the whole tower to the belt: r2c5 empties, the yellow key flies and opens the
    yellow lock (key-unlock credit)."""
    res = analyzer.analyze(shot("054"), log=lambda s: None)
    cell = res.view.grid[2][5]
    assert (cell.color, cell.body, cell.face_up) == ("darkgreen", "cyan", True)
    assert res.view.tappable() == [(2, 5)] and not res.view.disagreements
    move = res.decision.move
    assert move.pos == (2, 5) and move.color == "darkgreen"
    assert move.unlocks == [("yellow", ((2, 1),))]
    assert move.stages == ["cyan"] and move.short_stages == []   # plenty of cyan sand
    assert "opens lock yellow with 1 bucket(s)" in move.describe()
    nxt = res.state.after_tap((2, 5))
    assert nxt.cell((2, 5)).kind is Kind.EMPTY and nxt.cell((2, 6)).kind is Kind.EMPTY
    assert nxt.conveyor[-1].stages == ("cyan",)


def test_tower_with_a_starving_later_stage_is_penalised():
    st = make_state(["red/blue red"], sand={"red": 0.4, "blue": 0.001},
                    exposed={"red": 0.3})
    d = choose(st, SC)
    tower, plain = by_pos(d)[(0, 0)], by_pos(d)[(0, 1)]
    assert tower.short_stages == ["blue"] and "blue (sand short)" in tower.describe()
    assert tower.score == pytest.approx(plain.score - SC.weight_tower_short_stage)
    assert d.move.pos == (0, 1)


def test_demand_aware_exposure_blocks_a_second_bucket():
    # contact layer 30% red; one empty red bucket on the belt claims 25% -> avail 5%
    st = make_state(["red blue"], k=1, conveyor=[ConveyorBucket("red", 0)],
                    sand={"red": .4, "blue": .4}, exposed={"red": .3, "blue": .1})
    d = choose(st, SC)
    assert by_pos(d)[(0, 0)].avail == pytest.approx(0.05)
    assert d.move.color == "blue"      # 10% avail, new colour, beats red's 5%
    st.exposed["red"] = .9             # clearly enough for both
    assert choose(st, SC).move.color == "red"


def test_stalled_belt_colour_waits_while_the_belt_collects():
    """One more bucket of a stalled colour is a lost slot: rejected while another belt bucket
    still collects (waiting helps)."""
    st = make_state(["red blue"], k=2, conveyor=[ConveyorBucket("red", 49),
                                                 ConveyorBucket("green", 20)],
                    sand={"red": .4, "blue": .4, "green": .2},
                    exposed={"red": .9, "blue": .1, "green": .3})
    st.belt_stalled = {"red": 10.0, "green": 0.5}
    d = choose(st, SC)
    red = by_pos(d)[(0, 0)]
    assert "lost slot" in red.rejected and "stalled 10s" in red.describe()
    assert d.move.color == "blue"
    st.belt_stalled = {"red": 1.0}     # below stall_s: still fine
    assert choose(st, SC).move.color == "red"


def test_stalled_belt_colour_is_a_penalised_last_resort_on_an_idle_belt():
    """Belt idle (nothing collects): waiting changes nothing. The stalled colour is allowed but
    loses weight_stalled_colour, so any other move goes first."""
    st = make_state(["red blue"], k=1, conveyor=[ConveyorBucket("red", 49)],
                    sand={"red": .4, "blue": .4}, exposed={"red": .9, "blue": .1})
    st.belt_stalled = {"red": 10.0}
    d = choose(st, SC)
    red = by_pos(d)[(0, 0)]
    assert red.rejected is None and red.stalled == 10.0
    assert d.move.color == "blue"
    alone = make_state(["red"], k=1, conveyor=[ConveyorBucket("red", 49)],
                       sand={"red": .4}, exposed={"red": .9})
    alone.belt_stalled = {"red": 10.0}
    d = choose(alone, SC)
    assert d.move.color == "red"
    assert d.move.score == pytest.approx(choose(make_state(
        ["red"], k=1, conveyor=[ConveyorBucket("red", 49)], sand={"red": .4},
        exposed={"red": .9}), SC).move.score - SC.weight_stalled_colour)


def test_path_avoids_a_stalled_colour_step():
    """Two 2-step paths to the useful face-down green: through cream (better exposed, but stalled
    on the belt) or through blue. With cream stalled the planner takes the blue one."""
    def state(stalled):
        st = make_state(["cream x blue",
                         "orange green_ orange"], k=1, conveyor=[ConveyorBucket("cream", 99)],
                        sand={"cream": .2, "blue": .2, "orange": .3, "green": .3},
                        exposed={"cream": .3, "blue": .2, "orange": .2, "green": .3})
        st.belt_stalled = stalled
        return st

    def green_plan(d):
        return next(p for p in d.plans if p.label.endswith("green") and p.rejected is None)

    assert green_plan(choose(state({}), SC)).steps[0] == (0, 0)
    d = choose(state({"cream": 10.0}), SC)
    assert green_plan(d).steps[0] == (0, 2)
    assert d.move.pos == (0, 2)


def test_tie_break_prefers_colour_not_on_the_belt():
    sc = StrategyConfig(weight_new_colour=0.0, belt_demand_exposure=0.0)
    st = make_state(["red blue"], k=1, conveyor=[ConveyorBucket("red", 50)],
                    sand={"red": .4, "blue": .4}, exposed={"red": .5, "blue": .5})
    assert choose(st, sc).move.color == "blue"


def test_belt_stall_across_frames(analyzer):  # noqa: F811
    """The same frame twice, 10 s apart: the red bucket's fill didn't grow -> stalled."""
    from analyzer import Analyzer

    an = Analyzer(analyzer.cfg)
    img = shot("052")
    assert an.analyze(img, log=lambda s: None, now=0.0).state.belt_stalled == {"red": 0.0}
    res = an.analyze(img, log=lambda s: None, now=10.0)
    assert res.state.belt_stalled == {"red": 10.0}
    assert by_pos(res.decision)[(3, 2)].stalled == 10.0


def test_non_gameplay_has_no_decision(analyzer):  # noqa: F811
    res = analyzer.analyze(shot("039"), log=lambda s: None)
    assert res.screen == "fail" and res.decision is None and res.tap_rel() is None


def test_draw_decision_marks_the_move(analyzer):  # noqa: F811
    from strategy import draw_decision

    img = shot("043")
    res = analyzer.analyze(img, log=lambda s: None)
    out = draw_decision(img, res.view, res.decision)
    assert out.shape == img.shape and not np.array_equal(out, img)


# --- overlay --dry-run -----------------------------------------------------------------------

def test_dry_run_analyses_only_stable_frames_and_annotates():
    pytest.importorskip("cv2")
    from analyzer import Analysis
    from tools.overlay import DryRun

    calls = []

    def fake_analyze(frame, log=print, diff=None):
        calls.append(frame.shape)
        return Analysis("menu")

    t = [0.0]
    dry = DryRun(fake_analyze, interval_s=0.5, stable_below=0.01, clock=lambda: t[0])
    frame = np.zeros((40, 20, 3), np.uint8)
    dry.feed(frame, diff=0.5)                 # animating: skipped
    assert calls == [] and "waiting" in dry.status()
    dry.feed(frame, diff=0.0)
    dry._future.result()
    t[0] = 0.1
    dry.feed(frame, diff=0.0)                 # collects the result; too soon for another run
    assert calls == [(40, 20, 3)] and "screen: menu" in dry.status() and "read fails -" in dry.status()
    assert dry.annotate(frame) is frame       # nothing to draw outside gameplay
    dry.close()


def test_dry_run_tools_never_import_actions():
    code = ("import sys; import tools.overlay, strategy, state; "
            "assert 'actions' not in sys.modules, 'actions imported'")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


# --- failed reads and the decision gate ----------------------------------------------------

def _fresh_analyzer(analyzer, tmp_path, max_saves=300):  # noqa: F811
    import dataclasses

    from analyzer import Analyzer

    cfg = dataclasses.replace(analyzer.cfg, paths=dataclasses.replace(analyzer.cfg.paths,
                                                                       debug=str(tmp_path)),
                              strategy=dataclasses.replace(analyzer.cfg.strategy,
                                                           vision_fail_max_saves=max_saves))
    return Analyzer(cfg)


def _broken(kind):
    img = shot("052").copy()
    h, w = img.shape[:2]
    if kind == "grid":       # grid area painted with the panel colour
        img[int(.55 * h):int(.85 * h)] = img[int(.6 * h), 5]
    else:                    # the k digit of "k/5" painted over
        img[int(.51 * h):int(.545 * h), int(.42 * w):int(.49 * w)] = img[int(.53 * h), int(.43 * w)]
    return img


def test_failed_reads_are_saved_with_step_and_counted(analyzer, tmp_path):  # noqa: F811
    import json

    an = _fresh_analyzer(analyzer, tmp_path)
    ok = an.analyze(shot("052"), log=lambda s: None)
    assert ok.failure is None and (ok.reads, ok.fails) == (1, 0)
    g = an.analyze(_broken("grid"), log=lambda s: None, diff=0.002)
    assert g.screen == "gameplay" and g.failure.startswith("grid: ") and g.decision is None
    c = an.analyze(_broken("counter"), log=lambda s: None)
    assert c.failure == "counter: k/5 not read" and c.decision.move is None
    assert (c.reads, c.fails) == (3, 2)
    recs = [json.loads(l) for l in (tmp_path / "vision_fail" / "fails.jsonl").read_text().splitlines()]
    assert [r["step"] for r in recs] == ["grid", "counter"] and recs[0]["diff"] == 0.002
    assert all((tmp_path / "vision_fail" / r["frame"]).exists() for r in recs)
    assert an.analyze(shot("039"), log=lambda s: None).reads == 3   # not gameplay: not counted


def test_failed_read_saves_are_capped(analyzer, tmp_path):  # noqa: F811
    an = _fresh_analyzer(analyzer, tmp_path, max_saves=1)
    logs = []
    for _ in range(3):
        an.analyze(_broken("counter"), log=logs.append)
    assert len(list((tmp_path / "vision_fail").glob("*.png"))) == 1 and an.fails == 3
    assert any("not saving more" in l for l in logs)


def test_decision_gate_needs_consecutive_agreeing_reads():
    from analyzer import Analysis, DecisionGate
    from strategy import Candidate, Decision

    def res(pos=None, color="red", failure=None):
        move = None if pos is None else Candidate(pos, color)
        return Analysis("gameplay", decision=Decision(move, "x"), failure=failure)

    gate = DecisionGate(2)
    assert gate.feed(res((0, 1))) is None                       # 1/2
    assert gate.feed(res((0, 1))).pos == (0, 1)                 # 2/2: confirmed
    assert gate.feed(res((0, 1))).pos == (0, 1)                 # still agreeing
    assert gate.feed(res((0, 2))) is None and gate.count == 1   # changed its mind: restart
    assert gate.feed(res((0, 2), failure="counter: k/5 not read")) is None and gate.count == 0
    assert gate.feed(res((0, 2))) is None                       # failure broke the streak
    assert gate.feed(res((0, 2), color="blue")) is None         # same cell, other colour
    assert gate.feed(res(None)) is None and gate.count == 0     # wait resets
    gate.feed(res((1, 1)))
    gate.reset()                                                # after a tap
    assert gate.feed(res((1, 1))) is None


def test_dry_run_status_shows_confirmation_and_fail_rate(analyzer, tmp_path):  # noqa: F811
    from tools.overlay import DryRun

    an = _fresh_analyzer(analyzer, tmp_path)
    dry = DryRun(an.analyze, interval_s=0, stable_below=1.0, confirm_frames=2)

    def step(img):
        dry.feed(img, diff=0.0)
        dry._future.result()
        dry.feed(img, diff=1.0)       # collect, no new analysis

    step(shot("052"))
    assert "pending 1/2" in dry.status() and dry.confirmed is None
    step(shot("052"))
    assert "CONFIRMED 2/2" in dry.status() and dry.confirmed.pos == (2, 3)
    step(shot("052"))                 # still agreeing: same text, so it is printed only once
    assert "CONFIRMED 2/2" in dry.status() and dry.gate.count == 3
    step(_broken("counter"))
    assert dry.confirmed is None and "read failed (counter" in dry.status()
    assert "read fails 1/4 (25%)" in dry.status()
    dry.close()


def test_buckets_that_are_not_candidates_show_why():
    st = make_state(["red", "blue_"], k=0, sand={"red": .5, "blue": .5},
                    exposed={"red": .5, "blue": .5})
    st.tappable, st.disagreements = [], [(0, 0)]   # red looks face-up, path rule disagrees
    st.grid[1][0].occluded = True                   # blue hidden by a hint arrow
    d = choose(st, SC)
    assert d.move is None and d.reason == "wait: no tappable bucket"
    assert "path rule says" in by_pos(d)[(0, 0)].rejected
    assert "hint arrow" in by_pos(d)[(1, 0)].rejected


def test_live_regression_056_cyan_reach_and_no_pointless_wait(analyzer):  # noqa: F811
    """L183, 1/5: the only tappable bucket is r1c1 cyan and it opens the way to more buckets.
    The 10 px contact layer read cyan 2.6 % (< 3 %) and the bot waited; a bucket also takes the
    grains a little higher up, where the sky is (reach band: ~15 %), and with 4 free slots
    there is no reason to wait anyway."""
    res = analyzer.analyze(shot("056"), log=lambda s: None)
    assert res.state.exposed["cyan"] > 0.1
    d = res.decision
    assert (d.move.pos, d.move.color) == ((1, 1), "cyan") and d.move.opens


# --- waiting must have a reason (GAME_RULES "When to wait") -----------------------------------

def low_exposure_state(k, conveyor=(), stalled=None):
    st = make_state(["red"], k=k, conveyor=list(conveyor), sand={"red": .5, "blue": .5},
                    exposed={"red": .02, "blue": .9})
    st.belt_stalled = dict(stalled or {})
    return st


def test_exposure_minimum_scales_with_free_slots():
    filling = [ConveyorBucket("blue", 40)] * 3
    assert "2 free slots" in choose(low_exposure_state(3, filling), SC).candidates[0].rejected
    for k in (0, 1, 2):     # 3+ free slots: no exposure needed, enough sand left is enough
        assert choose(low_exposure_state(k, filling[:k]), SC).move.pos == (0, 0)
    d = choose(low_exposure_state(4, filling + [ConveyorBucket("blue", 10)]), SC)
    assert d.move is None and "last slot" in d.candidates[0].rejected


def test_never_wait_when_no_belt_bucket_collects():
    """2 free slots, but the 3 blue buckets on the belt haven't filled for 10 s: waiting changes
    nothing, so the low-exposure red goes."""
    belt = [ConveyorBucket("blue", 40)] * 3
    assert choose(low_exposure_state(3, belt), SC).move is None
    d = choose(low_exposure_state(3, belt, stalled={"blue": 10.0}), SC)
    assert d.move is not None and d.move.pos == (0, 0)


def test_unseen_belt_bucket_counts_as_collecting():
    # k says 3 but only 2 buckets were read: the third may be collecting -> not idle
    belt = [ConveyorBucket("blue", 40)] * 2
    assert choose(low_exposure_state(3, belt, stalled={"blue": 10.0}), SC).move is None


def test_tap_that_opens_a_useful_bucket_may_go_with_low_exposure():
    st = make_state(["red", "darkred_"], k=3, conveyor=[ConveyorBucket("blue", 40)] * 3,
                    sand={"red": .3, "darkred": .3, "blue": .3},
                    exposed={"red": .01, "darkred": .5, "blue": .4})
    d = choose(st, SC)
    assert d.move is not None and d.move.pos == (0, 0)
    assert "r1c0 darkred (useful)" in d.move.describe()
    st.exposed["darkred"] = 0.0   # nothing useful behind it: the plain minimum applies
    assert choose(st, SC).move is None
