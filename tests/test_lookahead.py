"""Last-slot gamble (user rule 2026-10-20) and the phase-7 lookahead (lookahead.py), from live run
debug/runs/20261020-142035 (L187): the belt filled with buckets whose sand was buried under the
purple sky, then "no bucket passes the rules" for 50 s."""
import dataclasses

import pytest

from config import StrategyConfig
from lookahead import belt_outlook, outlook, settle
from state import ConveyorBucket
from strategy import choose, rule_check
from tests.test_state import analyzer, make_state  # noqa: F401  (fixture)

SC = StrategyConfig()
# the run's end: four stalled belt buckets, one of them out of the reader's sight (k=4, 3 seen),
# so the belt does not count as idle and the last slot needs 25 % exposure
STUCK_BELT = [ConveyorBucket("orange", 80), ConveyorBucket("blue", 64),
              ConveyorBucket("yellow", 58)]
STALLED = {"orange": 30.0, "blue": 50.0, "yellow": 48.0, "darkgreen": 32.0}
SAND = {"darkred": .08, "cream": .08, "orange": .06, "blue": .02, "yellow": .06,
        "darkgreen": .02, "purple": .05}


def stuck_state(row="darkred cream", exposed=None, stalled=STALLED, k=4):
    st = make_state([row], k=k, conveyor=STUCK_BELT, sand=dict(SAND),
                    exposed=exposed or {"darkred": .09, "cream": .06, "purple": .5})
    st.belt_stalled = dict(stalled)
    st.colour_capacity = {c: .04 for c in SAND}             # one bucket: 4 % of the picture
    st.above = {"darkred": {"orange": .09, "blue": .04, "darkred": .02},
                "cream": {"orange": .03, "purple": .01}}
    return st


def test_gamble_takes_the_last_slot_when_the_belt_is_stuck():
    d = choose(stuck_state(), SC)
    assert d.move is not None and d.move.gamble
    assert (d.move.pos, d.move.color) == ((0, 0), "darkred")   # uncovers 13 % for the belt
    assert d.reason.startswith("last-slot gamble: tap r0c0 darkred")
    assert d.move.uncovers == pytest.approx(.13)


def test_gamble_ties_go_to_the_bucket_s_own_exposure():
    st = stuck_state()
    st.above = {"darkred": {"orange": .05}, "cream": {"orange": .05}}
    st.exposed = {"darkred": .09, "cream": .12}
    assert choose(st, SC).move.color == "cream"


def test_no_gamble_while_a_belt_bucket_still_gains_sand():
    st = stuck_state(stalled={**STALLED, "orange": 3.0})   # orange grew 3 s ago
    d = choose(st, SC)
    assert d.move is None and "no bucket passes the rules" in d.reason


def test_the_25_percent_rule_holds_in_normal_cases():
    st = stuck_state(stalled={})                             # belt collecting
    d = choose(st, SC)
    assert d.move is None
    assert all("25%" in c.rejected for c in d.candidates)


def test_gamble_never_adds_a_stalled_colour_nor_one_that_cannot_fill():
    """Run 142035's end: yellow stalled on the belt (uncovers nothing). User (run 224631): a
    bucket whose colour has less sand left than a bucket takes never gets the last slot."""
    st = stuck_state(row="yellow darkred")
    st.exposed = {"yellow": .2, "darkred": .09, "purple": .5}
    st.above["yellow"] = {"orange": .5}
    st.sand["darkred"] = .001                                # darkred would never fill
    assert choose(st, SC).move is None
    only = stuck_state(row="yellow")                         # only the stalled yellow left
    only.exposed = {"yellow": .2, "purple": .5}
    assert choose(only, SC).move is None


def test_gamble_short_bucket_only_when_it_opens_one_that_fills():
    st = stuck_state(row="darkred")
    st.sand["darkred"] = .001
    assert choose(st, SC).move is None                       # opens nothing
    st = make_state(["darkred", "cream"], k=4, conveyor=STUCK_BELT, sand=dict(SAND),
                    exposed={"darkred": .09, "cream": .06, "purple": .5})
    st.belt_stalled, st.colour_capacity = dict(STALLED), {c: .04 for c in SAND}
    st.sand["darkred"] = .001
    d = choose(st, SC)
    assert d.move.gamble and d.move.color == "darkred" and d.move.fills == "opens r1c0 cream"
    assert "opens r1c0 cream" in d.reason


def test_gamble_prefers_a_bucket_that_fills_over_more_uncovered_sand():
    """Run 224631 #27: purple (1.95 % left, a bucket 3.67 %) uncovered more for the belt than
    yellow, which had ~1.6 buckets of sand: yellow must win."""
    st = stuck_state()                                       # darkred uncovers 13 %, cream 4 %
    st.sand["darkred"] = .03                                 # < one 4 % bucket
    d = choose(st, SC)
    assert d.move.gamble and d.move.color == "cream" and d.move.fills == "fills itself"


def test_gamble_needs_the_last_slot():
    st = stuck_state(k=3)                                    # two free: normal rules apply
    d = choose(st, SC)
    assert d.move is None or not d.move.gamble


# --- lookahead ---------------------------------------------------------------------------------

def sky_state():
    """Yellow starves on the belt under the purple sky. Orange is better exposed, but only
    purple's collection uncovers yellow."""
    st = make_state(["purple orange"], k=1, conveyor=[ConveyorBucket("yellow", 40)],
                    sand={"purple": .3, "orange": .2, "yellow": .2},
                    exposed={"orange": .35, "purple": .3})
    st.belt_stalled = {"yellow": 10.0}
    st.above = {"purple": {"yellow": .2, "purple": .1}, "orange": {"orange": .1}}
    return st


def test_settle_uncovers_what_lies_above_the_collecting_colours():
    st = sky_state().after_tap((0, 0))                       # purple on the belt, collecting
    contact = settle(st, SC)
    assert contact["yellow"] == pytest.approx(.2) and contact["purple"] == pytest.approx(.1)
    out = belt_outlook(st, SC)
    assert (out.live, out.dead, out.free) == (2, 0, 3)       # yellow lives again


def test_lookahead_prefers_the_colour_that_uncovers_the_starving_belt():
    flat = dataclasses.replace(SC, look_depth=0)
    assert choose(sky_state(), flat).move.color == "orange"  # greedy: best exposed
    d = choose(sky_state(), SC)
    assert d.move.color == "purple"
    assert d.move.look_bonus == pytest.approx(SC.weight_lookahead * 1 / 5)
    assert "lookahead r0c0 purple" in d.move.describe()


def test_lookahead_sees_two_taps_ahead():
    """Cream gives the belt nothing itself but opens purple; purple uncovers yellow."""
    st = make_state(["cream yellow_", "purple_ x"], k=1, conveyor=[ConveyorBucket("yellow", 40)],
                    sand={"purple": .3, "cream": .1, "yellow": .1},
                    exposed={"cream": .2, "purple": .3})
    st.belt_stalled = {"yellow": 10.0}
    st.above = {"purple": {"yellow": .2}}
    one = outlook(st, (0, 0), rule_check(dataclasses.replace(SC, look_depth=1)),
                  dataclasses.replace(SC, look_depth=1))
    three = outlook(st, (0, 0), rule_check(SC), SC)
    assert [c for _, c in three.steps] == ["cream", "purple"]
    assert three.value > one.value


def test_lookahead_rejects_a_tap_that_jams_a_collecting_belt():
    """Last slot: red passes the release rules (sand >= 0.5 x capacity) but cannot fill
    (< 1.0 x capacity; the purple sky has no buckets left). With green still collecting the
    tap is fine; with green short of sand as well the belt would be full with nothing live."""
    belt = [ConveyorBucket("blue", 50)] * 3 + [ConveyorBucket("green", 10)]
    st = make_state(["red"], k=4, conveyor=belt, exposed={"red": .5, "green": .2},
                    sand={"red": .1, "blue": .001, "green": .2, "purple": .3})
    red = next(c for c in choose(st, SC).candidates if c.pos == (0, 0))
    assert red.rejected is None
    st.sand = {"red": .1, "blue": .001, "green": .001, "purple": .5}
    red = next(c for c in choose(st, SC).candidates if c.pos == (0, 0))
    assert red.rejected and red.rejected.startswith("lookahead: the belt jams")


# --- a tapped bucket flying over empty cells (Q2 of the run) -----------------------------------

def test_flying_bucket_over_empty_cells_is_not_a_bucket():
    """Run 20261020-142035, 76.5 s: tap #20 r5c2 orange; one frame later r4c3 / r3c3 (empty
    before) read orange-up / cream/orange-up and cut the path of r3c2 / r4c2."""
    import numpy as np

    from cells import Kind
    from hints import InFlight
    from vision import Lattice

    lat = Lattice(101.0, 748.0, 56.6, 50.1, np.ones((3, 2), bool))
    before = make_state([". darkred", ". blue", "orange ."]).grid
    flight = make_state([". darkred", "orange blue", "orange ."]).grid   # r1c0 phantom
    logs = []
    f = InFlight(flight_s=2.0)
    assert f.apply(lat, before, now=0.0, log=logs.append) == []
    assert f.apply(lat, flight, now=1.0, log=logs.append) == [(1, 0)]
    assert flight[1][0].kind is Kind.EMPTY and "in flight" in logs[-1]
    late = make_state([". darkred", "orange blue", "orange ."]).grid
    assert f.apply(lat, late, now=2.5, log=logs.append) == []    # held 2.5 s: real (chest)
    assert late[1][0].kind is Kind.BUCKET
    f2 = InFlight(flight_s=2.0)
    f2.apply(lat, make_state([". darkred", ". blue", "orange ."]).grid, now=0.0)
    f2.pause(until=3.0)                                           # a chest opened
    landed = make_state([". darkred", "red blue", "orange ."]).grid
    assert f2.apply(lat, landed, now=1.0) == [] and landed[1][0].kind is Kind.BUCKET


# --- the run's own frames ---------------------------------------------------------------------

def run_state(analyzer, name, stalled):  # noqa: F811
    import cv2
    from pathlib import Path

    from state import LevelState
    from vision import read_gameplay

    img = cv2.imread(str(Path(__file__).resolve().parent / "data" / "run-20261020-142035"
                         / f"{name}.png"))
    gv = read_gameplay(img, analyzer.palette, analyzer.cfg.vision, analyzer.reader,
                       log=lambda s: None, icons=analyzer.icons, hints=analyzer.hints)
    return LevelState.from_view(gv, 5, analyzer.cfg.strategy.unknown_fill_need, stalled)


def test_run_end_gambles_on_the_move_that_opens_a_bucket_that_fills(analyzer):  # noqa: F811
    """00136 with the belt as at 118 s (every colour stalled >= 6 s, 4/5): the run waited 50 s.
    darkred r3c2 uncovers more for the belt (~6 % vs cream r1c6 ~4 %) but cannot fill (1.2 %
    sand left, a bucket 3.2 %); cream cannot either, but it opens purple r3c6, which can (the
    move the run's analysis called right: it drains the purple sky)."""
    pytest.importorskip("cv2")
    st = run_state(analyzer, "00136-wait",
                   {"yellow": 25.0, "orange": 6.0, "darkgreen": 9.0, "blue": 28.0})
    d = choose(st, analyzer.cfg.strategy)
    assert d.move is not None and d.move.gamble, d.reason
    assert (d.move.pos, d.move.color) == ((1, 6), "cream")
    assert d.move.fills == "opens r3c6 purple"
    assert d.reason.startswith("last-slot gamble")
    st.belt_stalled["orange"] = 1.0                          # orange still creeping up: wait
    assert choose(st, analyzer.cfg.strategy).move is None


def test_run_tap_21_blue_is_not_dead_with_the_whole_picture_read(analyzer):  # noqa: F811
    """Tap #21 / #22 (k=2, yellow + darkgreen stalled). Read only down to the zone bottom, blue
    r4c2 had 2.2 % sand (less than a bucket) and the lookahead counted it dead. The picture goes
    on below the zone (run 171133): read to its real bottom edge blue has 2.9 % - a bucket - and
    is as alive as orange."""
    pytest.importorskip("cv2")
    st = run_state(analyzer, "00108-tap", {"yellow": 2.0, "darkgreen": 31.0})
    assert st.sand["blue"] > 0.027
    d = choose(st, analyzer.cfg.strategy)
    by = {c.pos: c for c in d.candidates}
    assert by[(4, 2)].look_bonus == pytest.approx(by[(5, 4)].look_bonus)
