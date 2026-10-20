"""Regressions from the dry run debug/runs/20261017-103804 (L183, user playing, bot watching).

Frames in tests/data/run-20261017-103804/ (raw frames the run saved):
  278 lock:yellow at r2c1   -> 285 opened: orange under it, reads face-down (padlock animation)
  296 lock:red at r3c5      -> 304 opened: red under it, reads face-down
  413 belt cream 99% / red 87% / orange 30%, every tappable bucket is cream
"""
from pathlib import Path

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


@pytest.mark.parametrize("before, after, pos, colour", [
    # 00285 shows the yellow padlock body still fading out, not the bucket: "orange" was only the
    # nearest colour while the palette had no yellow (L187 added it: now it reads yellow). The
    # red case below reads the red padlock too (same colour by chance). Open question to the
    # user: how to treat a cell that still shows the opened lock's colour (docs/STATUS.md).
    pytest.param("00278-tap", "00285-wait", (2, 1), "orange", marks=pytest.mark.xfail(
        strict=True, reason="padlock body read, not the bucket under it (yellow since L187)")),
    ("00296-tap", "00304-tap", (3, 5), "red"),
])
def test_bucket_under_an_opened_lock_is_tappable_at_once(analyzer, before, after, pos,  # noqa: F811
                                                          colour):
    """The user tapped both right away: buckets under an opened lock are playable at once,
    although the padlock animation makes them read face-down. Path rule alone decides there."""
    from cells import Kind
    from hints import RecentUnlocks

    img0, img1 = frame(before), frame(after)
    alone = read(analyzer, img1)
    cell = alone.grid[pos[0]][pos[1]]
    assert cell.kind is Kind.BUCKET and cell.color == colour and not cell.face_up
    assert alone.playable[pos[0]][pos[1]]
    assert pos not in alone.tappable() and pos in alone.disagreements   # without the history

    unlocks = RecentUnlocks(trust_s=10.0)
    gv0 = read(analyzer, img0, unlocks, now=0.0)
    lock = gv0.grid[pos[0]][pos[1]]
    assert lock.kind is Kind.LOCK and lock.unlock == "key"
    gv1 = read(analyzer, img1, unlocks, now=4.0)
    assert pos in gv1.unlocked and pos in gv1.tappable() and pos not in gv1.disagreements

    late = read(analyzer, img1, unlocks, now=20.0)   # after trust_s: the face reading counts again
    assert pos not in late.unlocked and pos not in late.tappable()


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
