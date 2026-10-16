"""Greedy strategy: rules on synthetic states, decisions on saved shots, overlay dry-run."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from config import StrategyConfig
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
    st = make_state(["red blue"], k=0, sand={"red": .3, "blue": .3}, exposed={"blue": .5})
    d = choose(st, SC)
    assert d.move.pos == (0, 1)
    assert "not exposed" in by_pos(d)[(0, 0)].rejected


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
    ("043", (3, 2)),     # last slot: red is 67% of the bottom, opens two buckets
    ("008", (4, 0)),     # blue is the most exposed colour
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
        assert "not exposed" in by_pos(d)[pos].rejected


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
    from strategy import Analysis
    from tools.overlay import DryRun

    calls = []

    def fake_analyze(frame, log=print):
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
    assert calls == [(40, 20, 3)] and dry.status() == "dry-run: menu"
    assert dry.annotate(frame) is frame       # nothing to draw outside gameplay
    dry.close()


def test_dry_run_tools_never_import_actions():
    code = ("import sys; import tools.overlay, strategy, state; "
            "assert 'actions' not in sys.modules, 'actions imported'")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
