"""Regressions from the dry run debug/runs/20261017-103804 (L183, user playing, bot watching).

Frames in tests/data/run-20261017-103804/ (raw frames the run saved):
  278 lock:yellow at r2c1   -> 285 opened: orange under it, reads face-down (padlock animation)
  296 lock:red at r3c5      -> 304 opened: red under it, reads face-down
  413 belt cream 99% / red 87% / orange 30%, every tappable bucket is cream
"""
from pathlib import Path

import numpy as np
import pytest

from state import LevelState
from strategy import choose
from tests.test_state import analyzer  # noqa: F401  (fixture)

DATA = Path(__file__).resolve().parent / "data" / "run-20261017-103804"


def frame(name):
    import cv2

    img = cv2.imread(str(DATA / f"{name}.png"))
    assert img is not None, name
    return img


def read(an, img, unlocks=None, now=0.0):
    from vision import read_gameplay

    return read_gameplay(img, an.palette, an.cfg.vision, an.reader, log=lambda s: None,
                         icons=an.icons, hints=an.hints, unlocks=unlocks, now=now)


@pytest.mark.parametrize("before, after, pos, lock_colour", [
    ("00278-tap", "00285-wait", (2, 1), "yellow"),
    ("00296-tap", "00304-tap", (3, 5), "red"),
])
def test_opened_lock_waits_for_a_confident_colour(analyzer, before, after, pos,  # noqa: F811
                                                  lock_colour):
    """Buckets under an opened lock are playable at once (user, 2026-10-19), but right after
    opening the cell still shows the padlock fading out: both frames read the LOCK's colour
    (00285 used to read "orange" only because yellow was missing from the palette). User rule
    (2026-10-20): tap at once only with a confident colour, else wait until the padlock is gone
    (vision.unlock_wait_s); the path rule alone decides the face for unlock_trust_s."""
    from cells import Kind
    from hints import RecentUnlocks

    v = analyzer.cfg.vision
    img0, img1 = frame(before), frame(after)
    alone = read(analyzer, img1)
    cell = alone.grid[pos[0]][pos[1]]
    assert cell.kind is Kind.BUCKET and cell.color == lock_colour and not cell.face_up
    assert alone.playable[pos[0]][pos[1]]
    assert pos not in alone.tappable() and pos in alone.disagreements   # without the history

    unlocks = RecentUnlocks(trust_s=v.unlock_trust_s)
    gv0 = read(analyzer, img0, unlocks, now=0.0)
    lock = gv0.grid[pos[0]][pos[1]]
    assert lock.kind is Kind.LOCK and lock.unlock == "key" and lock.color == lock_colour
    gv1 = read(analyzer, img1, unlocks, now=4.0)        # opens here: reads the padlock's colour
    assert pos in gv1.unlocked and pos in gv1.unsure and pos not in gv1.tappable()
    assert pos not in gv1.disagreements
    t = 4.0 + v.unlock_wait_s + 0.1                     # padlock gone: the reading counts
    gv2 = read(analyzer, img1, unlocks, now=t)
    assert pos in gv2.unlocked and pos not in gv2.unsure and pos in gv2.tappable()

    late = read(analyzer, img1, unlocks, now=20.0)   # after trust_s: the face reading counts again
    assert pos not in late.unlocked and pos not in late.tappable()


def test_unlock_unsure_rules(analyzer):  # noqa: F811
    from cells import Cell, Kind
    from vision import unlock_unsure

    pal, v = analyzer.palette, analyzer.cfg.vision
    orange = tuple(pal.buckets["orange"][0])
    clear = Cell(2, 1, Kind.BUCKET, color="orange", face_up=False, seen_lab=orange)
    assert unlock_unsure(clear, "yellow", 0.1, pal, v) is None          # confident: tap at once
    assert "lock's colour" in unlock_unsure(clear, "orange", 0.1, pal, v)
    assert unlock_unsure(clear, "orange", v.unlock_wait_s, pal, v) is None   # waited long enough
    blurred = Cell(2, 1, Kind.BUCKET, color="orange", face_up=False,
                   seen_lab=tuple(np.add(orange, (0.0, 0.0, 15.0))))
    assert "not clear" in unlock_unsure(blurred, "yellow", 0.1, pal, v)


def end_state(an, stalled):
    gv = read(an, frame("00413-tap"))
    sc = an.cfg.strategy
    st = LevelState.from_view(gv, sc.conveyor_slots, sc.unknown_fill_need)
    st.belt_stalled = stalled
    return st


def test_no_more_cream_while_the_belt_still_collects(analyzer):  # noqa: F811
    """The run released cream (r4c5, a path step) although cream sat at 99% on the belt: the
    stall clock kept restarting. With cream stalled and red still collecting, no cream goes."""
    st = end_state(analyzer, {"cream": 10.0, "red": 0.5, "orange": 10.0})
    creams = [p for p in st.tappable if st.cell(p).color == "cream"]
    assert creams and len(creams) == len(st.tappable)
    d = choose(st, analyzer.cfg.strategy)
    assert d.move is None
    for c in d.candidates:
        if c.pos in creams:
            assert "lost slot" in c.rejected


def test_stalled_cream_is_a_penalised_last_resort_on_an_idle_belt(analyzer):  # noqa: F811
    """Every belt colour stalled: waiting changes nothing, and r4c5 is the only way on (it opens
    the path to r5c6 darkgreen). It may go, with the stalled-colour penalty."""
    sc = analyzer.cfg.strategy
    st = end_state(analyzer, {"cream": 10.0, "red": 10.0, "orange": 10.0})
    d = choose(st, sc)
    assert d.move is not None and d.move.color == "cream" and d.move.stalled == 10.0
    free = choose(end_state(analyzer, {"red": 10.0, "orange": 10.0}), sc).move
    assert free is not None and d.move.score < free.score - sc.weight_stalled_colour / 2
