"""Unlock value and path planning (planner.py) on small synthetic grids."""
import dataclasses

from config import StrategyConfig
from planner import plan, shortest_path, release_costs, usefulness
from strategy import choose, path_check
from tests.test_state import make_state

SC = StrategyConfig()
NO_PATHS = dataclasses.replace(SC, weight_path_bonus=0.0)


def by_pos(decision):
    return {c.pos: c for c in decision.candidates}


def path_level(k=0, cream_exposed=.35, cream_sand=.3):
    """Greedy takes orange (most exposed, opens another useful orange). The right move is cream:
    two creams wall off a darkred bucket while darkred fills the lower part of the picture."""
    return make_state(["orange cream",
                       "orange_ cream_",
                       "green_ darkred_"], k=k,
                      sand={"orange": .3, "cream": cream_sand, "darkred": .3, "green": .01},
                      exposed={"orange": .45, "cream": cream_exposed, "darkred": .05},
                      lower={"darkred": .6, "orange": .3})


def test_usefulness_counts_lower_sand_and_belt_demand():
    st = path_level()
    assert usefulness(st, "darkred", SC) == .3             # max(.05, .5 x .6)
    assert usefulness(st, "green", SC) == 0.0              # no sand of it near the bottom
    after = st.after_tap((0, 0))                          # an empty orange now claims .25
    assert abs(usefulness(after, "orange", SC) - .2) < 1e-9


def test_unlock_value_prefers_opening_a_useful_colour():
    st = make_state(["red blue",
                     "cream_ darkred_"], k=0, sand={"red": .3, "blue": .3, "cream": .2, "darkred": .2},
                    exposed={"red": .3, "blue": .3}, lower={"darkred": .6})
    d = choose(st, SC)
    assert d.move.pos == (0, 1)          # without the unlock value the tie goes left (red)
    assert "r0c1 blue -> opens r1c1 darkred (useful)" in d.move.describe()
    red = by_pos(d)[(0, 0)]
    assert "opens r1c0 cream" in red.describe() and "(useful)" not in red.describe()


def test_greedy_best_is_wrong_and_the_path_move_is_right():
    st = path_level()
    greedy = choose(st, NO_PATHS)
    assert greedy.move.pos == (0, 0) and greedy.move.color == "orange"
    d = choose(st, SC)
    assert d.move.pos == (0, 1) and d.move.color == "cream"
    best = d.plans[0]
    assert best.target == (2, 1) and best.steps == [(0, 1), (1, 1)] and best.rejected is None
    assert "path to r2c1 darkred: step 1/2" in d.reason
    assert "path to r2c1 darkred: step 1/2" in d.move.describe()


def test_path_avoids_buckets_that_are_unsafe_to_release():
    st = path_level()
    costs = release_costs(st, path_check(SC), SC)
    assert (2, 0) not in costs           # green: not enough sand left -> impassable
    cost, steps = shortest_path(st, (2, 1), costs)
    assert steps == [(0, 1), (1, 1)] and cost == 2.0


def test_no_path_bonus_when_a_step_would_deadlock_the_belt():
    # 4/5 on the belt: the first cream takes the last slot, the second has none -> never planned
    st = path_level(k=4)
    d = choose(st, SC)
    darkred = [p for p in d.plans if p.target == (2, 1)][0]
    assert darkred.rejected and "step 2" in darkred.rejected and "conveyor full" in darkred.rejected
    assert d.move.color == "orange" and d.move.plan is None


def test_no_path_bonus_when_the_second_step_starves():
    # cream barely exposed: the second empty cream on the belt would take sand that isn't there
    st = path_level(cream_exposed=.2)
    d = choose(st, SC)
    darkred = [p for p in d.plans if p.target == (2, 1)][0]
    assert darkred.rejected and "step 2" in darkred.rejected and "exposed" in darkred.rejected
    assert d.move.color == "orange"


def test_no_path_through_a_colour_short_of_sand():
    st = path_level(cream_sand=.02)
    assert (0, 1) not in release_costs(st, path_check(SC), SC)
    assert all(p.target != (2, 1) or p.rejected for p in plan(st, path_check(SC), SC))
    assert choose(st, SC).move.color == "orange"


def test_path_to_a_key():
    st = make_state(["orange cream .",
                     "orange_ cream_ x",
                     "green_ key:yellow lock:yellow"], k=0,
                    sand={"orange": .3, "cream": .3, "green": .1},
                    exposed={"orange": .45, "cream": .35}, lower={"orange": .3})
    assert choose(st, NO_PATHS).move.color == "orange"
    d = choose(st, SC)
    key = d.plans[0]
    assert key.label.startswith("key yellow (lock: ") and key.steps == [(0, 1), (1, 1)]
    assert d.move.pos == (0, 1) and "path to key yellow (lock: 0 bucket(s) behind): step 1/2" in d.reason


def test_one_step_targets_are_left_to_the_unlock_value():
    st = make_state(["red", "darkred_"], k=0, sand={"red": .3, "darkred": .3},
                    exposed={"red": .3}, lower={"darkred": .6})
    assert plan(st, path_check(SC), SC) == []
