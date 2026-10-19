"""L186 chests and mystery buckets in the state lookahead, planner and strategy (synthetic)."""
import pytest

from config import StrategyConfig
from planner import replay, scatter_risk
from strategy import choose, path_check
from tests.test_state import make_state

SC = StrategyConfig()
SAND = {"red": .3, "blue": .3, "green": .3}
EXPOSED = {"red": .3, "blue": .3, "green": .3}


def by_pos(decision):
    return {c.pos: c for c in decision.candidates}


def test_chest_and_mystery_count_as_hidden_buckets():
    st = make_state(["red chest4 mystery"])
    assert st.hidden_buckets == 4 + 1
    assert st.pending_buckets() == 1 + 5


def test_tap_that_connects_a_chest_opens_it():
    st = make_state(["red blue",
                     "chest3 green_"], sand=SAND, exposed=EXPOSED)
    after = st.after_tap((0, 0))
    assert after.chests_opened == [((1, 0), 3)]
    cell = after.cell((1, 0))
    assert cell.kind.value == "unknown" and cell.hidden == 3   # worst case: an obstacle
    assert after.hidden_buckets == st.hidden_buckets            # the 3 buckets still count
    assert st.after_tap((0, 1)).chests_opened == []


def test_scatter_risk_counts_free_cells_on_paths():
    st = make_state(["red blue x",
                     "chest2 . green",
                     "x . x"], sand=SAND, exposed=EXPOSED)
    after = st.after_tap((0, 0))                 # r0c0 empties: the chest connects and opens
    assert after.chests_opened == [((1, 0), 2)]
    risk = scatter_risk(after)
    # landing: r0c0, r1c1, r2c1 + the chest cell; green r1c2 hangs on r0c0, r1c0 and r1c1,
    # r2c1 is a dead end
    assert (risk.landing, risk.important, risk.buckets) == (4, 3, 2)
    assert risk.expected_cuts == 2 * 3 / 4


def test_chest_opening_is_penalised_but_not_forbidden():
    rows = ["red blue",
            "chest4 x"]
    st = make_state(rows, sand=SAND, exposed={"red": .3, "blue": .25})
    d = choose(st, SC)
    red = by_pos(d)[(0, 0)]
    assert red.chests == [((1, 0), 4)] and red.scatter is not None
    assert "opens chest4 r1c0" in red.describe()
    assert d.move.pos == (0, 1)                  # the other move goes first
    alone = make_state(["red", "chest4"], sand=SAND, exposed={"red": .3})
    d = choose(alone, SC)
    assert d.move.pos == (0, 0) and d.move.chests  # nothing better: open it


def test_few_free_cells_make_a_chest_cheaper():
    sc = StrategyConfig(weight_chest_cut=0.0)
    many = make_state(["red x x x", "chest4 . . ."], sand=SAND, exposed=EXPOSED)
    few = make_state(["red x x x", "chest4 green green green"], sand=SAND, exposed=EXPOSED)
    c_many, c_few = by_pos(choose(many, sc))[(0, 0)], by_pos(choose(few, sc))[(0, 0)]
    assert c_many.scatter.free_share == 1.0 and c_few.scatter.free_share == 2 / 5
    assert c_few.score > c_many.score


def test_plan_stops_at_a_chest_it_opens():
    # path to the useful face-down green r3c1: red r0c1, then blue r2c1. Tapping red connects
    # r1c1 and with it the chest r1c0: the board changes before the second step
    st = make_state(["x red x",
                     "chest2 . x",
                     "x blue x",
                     "x green_ x"], k=0, sand=SAND, exposed=EXPOSED)
    why = replay(st, [(0, 1), (2, 1)], path_check(SC))
    assert why is not None and "opens chest r1c0" in why
    assert replay(st, [(0, 1)], path_check(SC)) is None     # opening it at the end is fine


def test_tapping_next_to_a_mystery_bucket_reveals_it():
    st = make_state(["red blue",
                     "mystery x"], sand=SAND, exposed={"red": .3, "blue": .3})
    after = st.after_tap((0, 0))
    assert after.revealed == [(1, 0)] and (1, 0) in after.connected()
    d = choose(st, SC)
    red, blue = by_pos(d)[(0, 0)], by_pos(d)[(0, 1)]
    assert red.reveals == [(1, 0)] and red.reveals_reachable == [(1, 0)]
    assert "reveals r1c0 (reachable)" in red.describe()
    # a revealed neighbour of a tapped bucket is always connected right after the tap (the
    # tapped cell is an empty cell on the exit side), so both bonuses apply
    assert red.score - blue.score == pytest.approx(SC.weight_mystery_reveal
                                                   + SC.weight_mystery_reachable)
    assert d.move.pos == (0, 0)


def test_no_reveal_without_a_4_neighbour():
    st = make_state(["red blue",
                     "x mystery"], sand=SAND, exposed=EXPOSED)   # diagonal to red
    d = choose(st, SC)
    assert by_pos(d)[(0, 0)].reveals == [] and by_pos(d)[(0, 1)].reveals == [(1, 1)]
