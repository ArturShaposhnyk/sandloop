"""Regressions from the live run debug/runs/20261020-171133 (L187 --levels 3: lost twice, then
an ad stopped the session).

* Endgame: the belt filled with buckets at 96-99 % whose last sand lay under darkred / cream.
  Darkred r3c2 was tappable all along (r3c3 is empty), but "not enough sand (0.0 %)": the
  picture goes on ~12 px below the configured zone and darkred settles at the very bottom, so
  the zone read 0.17 % (dropped below the 0.2 % report limit) where the picture has 1.3 %.
  Plans that ended "5 dead on the belt" were followed.
* After Try Again full-screen ads without any known anchor (Trade Republic with a ▶| skip disc
  top right, Netflix) were "unknown screens"; the second one stopped the run after 61 s.

tests/data/run-20261020-171133/: 00535-tap (tap #53, k 3), 00551-tap (tap #54 yellow took the
last slot, k 4 with 3 buckets read), 00301 (the Trade Republic ad), 00587 (the Netflix ad).
"""
import dataclasses
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

from adclose import find_close, find_skip  # noqa: E402
from analyzer import colour_capacity  # noqa: E402
from flow import AD_SKIP, ScreenFlow, plan  # noqa: E402
from levelplan import Plan, PlanRunner, Step, _rank, plan_level  # noqa: E402
from screens import ScreenResult  # noqa: E402
from sim import Picture, Sim, SimConfig, UNREAD, grid_signature, picture_from_frame  # noqa: E402
from state import ConveyorBucket, LevelState  # noqa: E402
from strategy import choose  # noqa: E402
from tests.test_state import analyzer, make_state  # noqa: E402,F401  (fixture)
from config import StrategyConfig  # noqa: E402

DATA = Path(__file__).resolve().parent / "data" / "run-20261020-171133"
START = Path(__file__).resolve().parent / "data" / "run-20261020-145310" / "00002-tap.png"
STALLED = {"00535-tap": {"darkgreen": 0.0, "orange": 6.7, "blue": 3.5},
           "00551-tap": {"darkgreen": 8.3, "orange": 3.3, "blue": 14.8, "purple": 1.7}}


def frame(name):
    img = cv2.imread(str(DATA / f"{name}.png"))
    assert img is not None, name
    return img


def read(an, img):  # noqa: F811
    from vision import read_gameplay

    return read_gameplay(img, an.palette, an.cfg.vision, an.reader, log=lambda s: None,
                         icons=an.icons, hints=an.hints)


def run_state(an, name):  # noqa: F811
    start = LevelState.from_view(read(an, cv2.imread(str(START))))
    gv = read(an, frame(name))
    st = LevelState.from_view(gv, 5, an.cfg.strategy.unknown_fill_need, STALLED[name])
    st.colour_capacity = colour_capacity(start)
    return gv, st


# --- sand: the whole picture ----------------------------------------------------------------

def test_sand_is_read_down_to_the_picture_bottom(analyzer):  # noqa: F811
    _, st = run_state(analyzer, "00551-tap")
    assert st.sand["darkred"] > 0.01                    # was missing (0.17 % in the zone)
    assert st.sand["darkred"] >= st.capacity_for("darkred")


def test_endgame_taps_the_darkred_bucket(analyzer):  # noqa: F811
    """#54: the run tapped yellow r2c2 into the last slot (darkred "not enough sand 0.0 %")."""
    _, st = run_state(analyzer, "00551-tap")
    d = choose(st, analyzer.cfg.strategy)
    assert d.move is not None and (d.move.pos, d.move.color) == ((3, 2), "darkred")


# --- key colours ------------------------------------------------------------------------------

def key_board(free_k):
    """darkred (40 % of the contact layer, covering the blue on the belt) sits behind a yellow
    bucket; a purple bucket elsewhere."""
    st = make_state([". . .", "yellow x purple", "darkred x x"], k=free_k,
                    conveyor=[ConveyorBucket("blue", 90)] * free_k,
                    sand={"darkred": .05, "yellow": .05, "purple": .05, "blue": .01},
                    exposed={"darkred": .4, "yellow": .1, "purple": .15, "blue": .01})
    st.above = {"darkred": {"blue": .05}}
    return st


def test_key_colour_keeps_the_slots_to_reach_it():
    from endgame import key_reserve

    sc = StrategyConfig()
    st = key_board(3)                                   # 2 free: darkred needs yellow + itself
    why = key_reserve(st, (1, 2), sc)
    assert why is not None and why.startswith("keys: keep 2 slot(s) to reach darkred r2c0")
    assert "in front: r1c0 yellow" in why
    assert key_reserve(st, (1, 0), sc) is None          # yellow is on the way to darkred
    st = key_board(2)                                   # 3 free: purple leaves 2
    assert key_reserve(st, (1, 2), sc) is None
    d = choose(key_board(3), sc)
    purple = next(c for c in d.candidates if c.color == "purple")
    assert purple.rejected is not None and d.move is not None and d.move.color == "yellow"


# --- the planner ------------------------------------------------------------------------------

def test_plan_prefers_fewer_dead_buckets():
    """A node that collected more sand but holds a bucket that can never fill ranks below one
    without (run 171133 followed plans ending "5 dead on the belt")."""
    from config import PlannerConfig

    st = make_state(["a b"], k=0)
    s = Sim.start(st, Picture(["a", "b"], [[0, 0, 1]] * 4, contact=1), SimConfig(slots=3))
    dead = s.tap((0, 1))
    dead.pic = Picture(["a", "b"], [[0, 0]] * 4, 1)     # no b sand left: b can never fill
    dead.collected = 100
    alive = s.clone()
    assert dead.dead_slots() == 1 and alive.dead_slots() == 0
    assert _rank(alive, 10, PlannerConfig()) > _rank(dead, 10, PlannerConfig())


def test_unread_belt_buckets_keep_their_slots():
    st = make_state(["a"], k=4, conveyor=[ConveyorBucket("a", 50)] * 3)
    s = Sim.start(st, Picture(["a"], [[0] * 5] * 4, 1), SimConfig(slots=5))
    assert len(s.belt) == 4 and s.belt[-1].colour == UNREAD and s.free == 1
    assert s.dead_slots() == 0 and s.stuck_slots() == 0


def test_plan_from_551_counts_the_unread_bucket(analyzer):  # noqa: F811
    _, st = run_state(analyzer, "00551-tap")
    pic = picture_from_frame(frame("00551-tap"), analyzer.palette, analyzer.cfg.vision,
                             analyzer.cfg.planner.cell_px)
    plan_ = plan_level(st, pic, analyzer.cfg.planner)
    assert plan_.dead == 0
    assert all(s.belt_before >= st.conveyor_k for s in plan_.steps[:1])


def test_plan_step_that_breaks_a_reserve_is_not_tapped():
    cfg = __import__("config").load_config()
    st = key_board(3)
    runner = PlanRunner(cfg)
    runner.signature, runner.last_plan_t = "test", 0.0
    runner.plan = Plan([Step((1, 2), "purple", 3, grid_signature(st.grid))], "horizon", 0, 0, 0)
    runner.log = [{"t": 0.0}]
    logs = []
    fallback = choose(st, cfg.strategy)
    d = runner.decide(None, None, st, fallback, 1.0, logs.append)
    assert d is fallback and any("breaks a slot reserve: keys:" in s for s in logs)


# --- ads --------------------------------------------------------------------------------------

def test_skip_button_of_the_trade_republic_ad(analyzer):  # noqa: F811
    img = frame("00301-before-window-lost")
    (mark,) = find_skip(img, analyzer.cfg.flow)
    assert mark.center_rel[0] > 0.93 and mark.center_rel[1] < 0.1
    assert find_skip(cv2.flip(img, 1), analyzer.cfg.flow) == []   # the mute disc: never
    assert find_close(img, analyzer.cfg.flow) == []
    for name in ("00535-tap", "00551-tap", "00587-screen-other-wait"):
        assert find_skip(frame(name), analyzer.cfg.flow) == [], name


def test_unknown_screen_after_try_again_is_searched_for_skip(analyzer):  # noqa: F811
    fc = dataclasses.replace(analyzer.cfg.flow, confirm_frames=1, tap_cooldown_s=0.0)
    img = frame("00301-before-window-lost")
    other = ScreenResult("other", {})
    assert plan(other, img, 3, fc).kind == "wait"       # no ad context: just wait
    fl = ScreenFlow(fc)
    fl.last_tap = 100.0                                 # Try Again tapped at 100 s
    act = fl.feed(other, img, 3, 150.0)
    assert act.kind == "tap" and act.button == AD_SKIP
    fl = ScreenFlow(fc)
    fl.last_tap = 100.0
    assert fl.feed(other, img, 3, 100.0 + fc.ad_after_tap_s + 1).kind == "wait"


def test_unknown_screens_wait_three_minutes(analyzer):  # noqa: F811
    fc = analyzer.cfg.flow
    assert fc.unknown_max_s >= 180 and fc.ad_max_s >= 180 and fc.unknown_frame_s <= 10
