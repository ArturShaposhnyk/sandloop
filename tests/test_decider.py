"""Redesign phase C: belief.py (belt slots by identity, online correction), worlds.py (sampled
hidden buckets), decider.py (search decider behind ``decider: search``)."""
import dataclasses
import random

import pytest

from belief import BeltBelief
from cells import Kind
from config import DeciderConfig, from_dict, load_config
from decider import WAIT, SearchDecider, search, search_config
from sim import Picture, Sim, SimConfig
from state import ConveyorBucket
from tests.test_state import make_state
from worlds import find_unknowns, missing_colours, sample_bag, worlds

DC = DeciderConfig()


def belt(*items):
    return [ConveyorBucket(c, f, stages=tuple(s)) for c, f, *s in items]


# --- belief ------------------------------------------------------------------------------------

def test_a_fresh_bucket_that_fills_counts_as_gaining_at_once():
    b = BeltBelief(DC)
    b.tapped("red", (), 0.0)
    assert b.gaining(1.0)                           # just tapped (fresh_grace_s)
    b.update(belt(("red", 0)), 1, 1.0)
    assert not b.gaining(3.5)                       # grace over, no growth seen
    b.update(belt(("red", 9)), 1, 4.0)              # +9 points: really filling
    assert b.gaining(4.0)
    stalled, rates = b.corrections(4.0)
    assert stalled == [] and rates[0] == pytest.approx(9 / 4.0)


def test_a_slot_without_growth_is_stalled_and_its_colour_listed():
    b = BeltBelief(DC)
    b.tapped("cream", (), 0.0)
    for t in range(1, 9):
        b.update(belt(("cream", 72)), 1, float(t))
    assert not b.gaining(8.0)
    assert b.stalled_colours(8.0) == ["cream"]


def test_two_buckets_of_a_colour_read_alternately_are_no_growth():
    b = BeltBelief(DC)
    b.tapped("orange", (), 0.0)
    b.tapped("orange", (), 0.0)
    b.update(belt(("orange", 40), ("orange", 10)), 2, 0.5)
    for t in range(1, 12):                           # one or the other on screen
        b.update(belt(("orange", 40 if t % 2 else 10)), 2, 0.5 + t)
    assert len(b.slots) == 2
    assert b.stalled_colours(11.5) == ["orange"]


def test_the_counter_lagging_behind_a_tap_drops_no_slot():
    b = BeltBelief(DC)
    b.update([], 1, 0.0)                             # one slot, colour not read yet
    b.update(belt(("blue", 50)), 1, 10.0)
    b.tapped("red", (), 11.0)
    b.update(belt(("blue", 55)), 1, 12.0)            # stale k: the tap is not counted yet
    assert [s.colour for s in b.slots] == ["blue", "red"]
    b.update(belt(("blue", 60), ("red", 5)), 2, 13.0)
    assert len(b.slots) == 2


def test_a_tower_moves_to_its_next_stage_at_100():
    b = BeltBelief(DC)
    b.tapped("orange", ("cream",), 0.0)
    b.update(belt(("orange", 60, ["cream"])), 1, 2.0)
    b.update(belt(("orange", 100, ["cream"])), 1, 4.0)
    assert b.slots[0].colour == "cream" and b.gaining(4.0)
    assert b.buckets()[0].color == "cream"


def test_the_counter_falling_removes_the_finished_slot():
    b = BeltBelief(DC)
    b.tapped("red", (), 0.0)
    b.tapped("blue", (), 0.0)
    b.update(belt(("red", 100), ("blue", 30)), 2, 5.0)
    b.update(belt(("blue", 40)), 1, 6.0)
    assert [s.colour for s in b.slots] == ["blue"]


# --- simulator hooks -----------------------------------------------------------------------------

def pic(*cols, contact=2, names=("red", "blue", "cream")):
    return Picture(list(names), [list(c) for c in cols], contact)


def one_slot_sim(colour_idx_cols, **kw):
    st = make_state(["."], k=1, conveyor=[ConveyorBucket("red", 0)])
    s = Sim.start(st, pic(*colour_idx_cols), SimConfig(fill_rate=100, fill_min_exposed=0.3,
                                                       fill_max=50), {"red": 4})
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_a_stalled_colour_gets_only_sand_uncovered_later():
    s = one_slot_sim([[0, 0, 1, 1]] * 3)              # red at the bottom: 100 % of contact
    s.offsets = {"red": 1.0}
    nxt = s.clone()
    for _ in range(5):
        nxt.lap()
    assert nxt.belt[0].left == 4                      # nothing reachable for red


def test_an_observed_rate_replaces_the_curve_and_reaches_deeper():
    s = one_slot_sim([[1, 1, 0, 0]] * 3)              # red right above the contact layer
    assert s.clone().lap() == 0                        # the curve: no red at the bottom
    s.belt[0].rate, s.belt[0].rate_until = 50.0, 10.0
    nxt = s.clone()
    for _ in range(3):
        nxt.lap()
    assert not nxt.belt                                # filled from 2 x contact deep


def test_shell_buckets_may_go_any_time_in_the_search_only():
    st = make_state(["red+shell blue"], k=0, sand={"red": 0.1, "blue": 0.1})
    p = pic([1, 1, 0], [1, 0, 0])
    strict = Sim.start(st, p, SimConfig())
    loose = Sim.start(st, p, dataclasses.replace(SimConfig(), strict_shells=False))
    assert (0, 0) not in strict.moves() and (0, 0) in loose.moves()


# --- sampled worlds ----------------------------------------------------------------------------

def test_hidden_buckets_follow_the_colour_balance():
    p = pic([0] * 30 + [1] * 10, [0] * 30 + [1] * 10)
    miss = missing_colours(p, {"red": 2.0, "blue": 1.0}, 3)
    assert miss["red"] > miss["blue"]                # red: 60 cells, 2 known; blue 20 / 1
    bag = sample_bag({"red": 2.0, "blue": 1.0}, 3, random.Random(1))
    assert sorted(bag) == ["blue", "red", "red"]


def test_a_tunnel_drops_its_sampled_queue_and_memory_is_kept():
    st = make_state(["red blue", "tunnel2:up ice1"], k=0)
    p = pic([0] * 20, [1] * 20, [2] * 20)
    unk = find_unknowns(st, [])
    assert unk.cells == [(1, 1)] and unk.tunnels == {(1, 0): 2}
    ws = worlds(st, p, SimConfig(), 3, 7, memory={(1, 1): ("cream", None)})
    assert len(ws) == 3
    for w in ws:
        q = w.tunnel_q[(1, 0)][1]
        s = w.tap((0, 0))                              # red taken: the tunnel drops q[0]
        assert s.grid.cell((0, 0)).kind is Kind.BUCKET and s.grid.cell((0, 0)).color == q[0]
        assert w.hidden[(1, 1)] == ("cream", None)     # seen before: not sampled
        s = s.tap((0, 1))                              # ice 1 -> 0: cream under it
        assert s.grid.cell((1, 1)).color == "cream"


def test_a_board_without_hidden_buckets_is_one_world():
    st = make_state(["red blue"], k=0)
    assert len(worlds(st, pic([0] * 9, [1] * 9), SimConfig(), 4, 0)) == 1


def test_a_chest_scatters_its_sampled_buckets_into_free_cells():
    st = make_state([". red", "chest2 ."], k=0)
    ws = worlds(st, pic([0] * 9, [1] * 9, [2] * 9), SimConfig(), 1, 3)
    s = ws[0].tap((0, 1))
    cells = [c for c in s.grid.cells(Kind.BUCKET)]
    assert len(cells) == 2 and all(c.color in ("red", "blue", "cream") for c in cells)


# --- the decider -------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c.decider = dataclasses.replace(c.decider, mode="search", budget_s=0.2,
                                    start_budget_s=0.2, max_depth=12)
    return c


def board(k=0, conveyor=()):
    return make_state(["red blue cream", "red_ blue_ cream_"], k=k, conveyor=list(conveyor),
                      sand={"red": 0.2, "blue": 0.2, "cream": 0.2},
                      exposed={"red": 0.6, "blue": 0.3, "cream": 0.1})


def board_pic():
    cols = [[0, 0, 1, 2] * 6] * 6 + [[1, 0, 2, 1] * 6] * 4
    return pic(*cols, contact=3)


def test_config_switch_short_form():
    assert from_dict({"decider": "search"}).decider.mode == "search"
    assert from_dict({}).decider.mode == "old"


def test_never_idle_while_no_belt_bucket_gains(cfg):
    d = SearchDecider(cfg)
    st = board(k=1, conveyor=[ConveyorBucket("blue", 40)])
    for t in range(0, 12, 2):                        # blue sits at 40 %
        d.observe(st, float(t))
    dec = d.decide(st, board_pic, 12.0)
    assert dec.move is not None and "no belt bucket is gaining" in dec.reason
    assert WAIT not in d.last.result.values


def test_wait_is_a_candidate_while_a_bucket_gains(cfg):
    d = SearchDecider(cfg)
    for t in range(0, 6, 2):
        st = board(k=1, conveyor=[ConveyorBucket("blue", 10 + 10 * t)])
        d.observe(st, float(t))
    d.decide(st, board_pic, 4.0)
    assert WAIT in d.last.result.values


def test_only_frame_tappable_buckets_are_first_taps(cfg):
    d = SearchDecider(cfg)
    st = board()
    st.tappable = [(0, 1)]                            # perception guards removed the others
    d.observe(st, 0.0)
    dec = d.decide(st, board_pic, 0.0)
    assert dec.move.pos == (0, 1)
    assert {a[1] for a in d.last.result.values if a != WAIT} == {(0, 1)}


def test_full_belt_or_unread_counter_waits(cfg):
    d = SearchDecider(cfg)
    st = board(k=5, conveyor=[ConveyorBucket("red", 10)] * 5)
    d.observe(st, 0.0)
    assert d.decide(st, board_pic, 0.0).move is None
    st.conveyor_k = None
    assert "counter" in d.decide(st, board_pic, 0.0).reason


def test_search_prefers_the_exposed_colour(cfg):
    sc = search_config(cfg, 5, False)
    st = board()
    roots = worlds(st, board_pic(), sc, 1, 0)
    one = dataclasses.replace(cfg.decider, max_depth=1)   # one tap: what it collects
    res = search(roots, [("tap", p) for p in roots[0].moves()], one, 0.3)
    assert res.ranked()[0][0] == ("tap", (0, 0))     # red: 60 % of the contact layer


def test_the_last_line_is_kept_when_the_beam_misses_it(cfg):
    d = SearchDecider(cfg)
    st = board()
    d.observe(st, 0.0)
    first = d.decide(st, board_pic, 0.0)
    line = d.line
    assert line and line[0] == ("tap", first.move.pos)
    d.tapped(first.move.pos, first.move.color, (), 0.0)
    assert d.line == line[1:]


def test_bot_plays_a_search_decision_on_a_real_frame(tmp_path):
    """decider: search end to end: Analyzer -> belief -> search -> gate -> tap (L183 shot 056:
    only r1c1 cyan is tappable)."""
    pytest.importorskip("cv2")
    from analyzer import Analyzer
    from runlog import load_events
    from tests.test_bot import S039, S056, Tapper, make_bot, run, shot

    c = load_config()
    c.bot = dataclasses.replace(c.bot, settle_min_s=0.0, settle_max_s=0.0, retap_guard_s=0.0,
                                speed_x2=False)
    c.decider = dataclasses.replace(c.decider, mode="search", budget_s=0.2, start_budget_s=0.2)
    an = Analyzer(c, clock=lambda: 0.0)
    g, f = shot(S056), shot(S039)
    tapper = Tapper()
    bot, log = make_bot(c, an, tmp_path, [g, g, f, f, f], tapper=tapper)
    run(bot, log)
    taps = [e for e in load_events(log.dir) if e["type"] == "tap"]
    assert len(tapper.taps) == 1 and taps[0]["pos"] == [1, 1]
    assert taps[0]["why"].startswith("confirmed, search 1/")
    assert taps[0]["decider"] == "search"
    assert taps[0]["decision"].startswith("tap r1c1 cyan (search ") and "belt:" in taps[0]["decision"]
    assert an.planner is not None   # the old decider runs alongside (all-lost fallback)


def test_every_candidate_lost_falls_back_to_the_old_decider(cfg):
    from strategy import Decision

    old = Decision(None, "wait: old decider")
    st = board()
    for limit, expect in ((0.0, True), (1.01, False)):
        d = SearchDecider(dataclasses.replace(cfg, decider=dataclasses.replace(
            cfg.decider, fallback_lost=limit)))
        d.observe(st, 0.0)
        dec = d.decide(st, board_pic, 0.0, fallback=lambda: old)
        assert (dec.reason.startswith("wait: old decider [old decider: every search candidate")
                == expect)
        assert d.fallbacks == int(expect)
