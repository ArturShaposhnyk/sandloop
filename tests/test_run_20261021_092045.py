"""Regressions from the live run debug/runs/20261021-092045 (L189, 14 taps, stopped by the user
after 200 s on a lost board).

The user's analysis: the belt was 4/5 (red 23, yellow 27, darkred 8, red 57) while the bottom
of the picture is cyan / cream; the cyan bucket that was needed is under the ice "1" at r3c1.
Breaking the ice takes one more tap, and that tap fills the belt: no winning move was left. The
mistake came earlier - the bot had to keep 2 free slots (1 for the tap that breaks the ice, 1
for the bucket under it).

1. The ice "1" read as a face-down cream/cyan tower (its thin digit has 0.040 dark pixels <
   ``ice_min_dark`` 0.05), so "STUCK ... misread grid?" for 200 s; the cyan bucket r4c6 read as
   ice without a number. Ice now needs its number (``counter: required``), dark minimum 0.03.
2. Ice reserve (``endgame.ice_reserve``): taps #12 red, #13 / #14 darkred (3-6 % of the contact
   layer, they never filled) are rejected while the ice still needs taps.
3. Lost position (``endgame.lost_finish``): belt stuck ``lost_stuck_s`` and nothing passes ->
   tap the best remaining bucket to end the level (then Try Again).
4. Booster hint arrows over the bottom row matched 0.60-0.79 (one frame none >= 0.8): r5c2
   (chain padlock) read "two objects", r5c6 (yellow key) read empty. A row of arrows at one
   height / scale is now masked (``hints._in_row``).
"""
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

from tests.test_bot import analyzer, cfg  # noqa: E402,F401  (fixtures)

DATA = Path(__file__).resolve().parent / "data" / "run-20261021-092045"
ROW5 = ["key:magenta", "red-down", "chain:yellow", "cyan-down", "orange-down", "x", "key:yellow"]


def frame(name):
    img = cv2.imread(str(DATA / f"{name}.png"))
    assert img is not None, name
    return img


def state_of(analyzer, name):  # noqa: F811
    return analyzer.analyze(frame(name), log=lambda s: None, now=0.0).state


@pytest.mark.parametrize("name,counter", [("00078-tap", 5), ("00094-tap", 3),
                                          ("00238-stuck-disagree", 1)])
def test_ice_reads_with_its_number(analyzer, name, counter):  # noqa: F811
    st = state_of(analyzer, name)
    ice = st.cell((3, 1))
    assert (ice.lock, ice.counter) == ("ice", counter)
    assert st.cell((4, 6)).label() == "cyan-down"        # a cyan bucket is no ice
    assert not st.disagreements


@pytest.mark.parametrize("name", ["00083-tap", "00094-tap"])
def test_ice_reserve_rejects_taps_that_stay_on_the_belt(analyzer, name):  # noqa: F811
    """#12 (ice 4, 3 free: red 3 %) and #13 (ice 3, 3 free: darkred 6 %) - every bucket on the
    path or not would sit on the belt while the ice still needs taps: wait."""
    from strategy import choose

    st = state_of(analyzer, name)
    d = choose(st, analyzer.cfg.strategy)
    assert d.move is None
    reasons = [c.rejected for c in d.candidates]
    assert any(r and r.startswith("ice r3c1") for r in reasons)
    assert all(r and ("ice r3c1" in r or "not enough exposed" in r) for r in reasons)


def test_ice_reserve_lets_a_filling_bucket_go(analyzer):  # noqa: F811
    from endgame import ice_reserve

    st = state_of(analyzer, "00083-tap")
    sc = analyzer.cfg.strategy
    assert ice_reserve(st, (0, 3), sc) is not None
    st.exposed["red"] = 0.3                               # it would fill and leave the belt
    assert ice_reserve(st, (0, 3), sc) is None


def test_ice_reserve_only_while_the_slots_exist(analyzer):  # noqa: F811
    """Ice 5 + 1 > 5 slots (frame 00078, before #11): nothing to keep."""
    from endgame import ice_reserve

    st = state_of(analyzer, "00078-tap")
    assert all(ice_reserve(st, p, analyzer.cfg.strategy) is None for p in st.tappable)


def test_lost_position_ends_the_level(analyzer):  # noqa: F811
    """The final board: ice 1, belt 4/5 of buckets that collect nothing. Stuck < lost_stuck_s:
    wait; after it: tap the red r2c2 (fills the belt, the level is lost -> Try Again)."""
    from strategy import choose

    st = state_of(analyzer, "00238-stuck-disagree")
    sc = analyzer.cfg.strategy
    st.belt_stalled = {c: sc.lost_stuck_s - 5 for c in ("yellow", "darkred", "red")}
    assert choose(st, sc).move is None
    st.belt_stalled = {c: sc.lost_stuck_s + 5 for c in ("yellow", "darkred", "red")}
    d = choose(st, sc)
    assert d.move is not None and d.move.lost and d.move.pos == (2, 2)
    assert d.reason.startswith("lost position: tap r2c2 red")


@pytest.mark.parametrize("name", ["unknown-092231-354", "unknown-092236-332"])
def test_booster_arrow_row_is_masked(analyzer, name):  # noqa: F811
    from hints import LatticeLock
    from vision import read_gameplay

    v = analyzer.cfg.vision
    lk = LatticeLock(v.lattice_lock_frames, v.lattice_min_fit)
    logs = []

    def read(img):
        return read_gameplay(img, analyzer.palette, v, analyzer.reader, log=logs.append,
                             icons=analyzer.icons, hints=analyzer.hints, lattices=lk)

    for _ in range(3):
        read(frame("00238-stuck-disagree"))
    gv = read(frame(name))
    assert "hint arrows: 4 masked out" in logs
    assert gv.labels()[5].split() == ROW5
    assert not gv.unknown_cells()


def test_bot_taps_the_lost_position_instead_of_hanging(cfg, tmp_path):  # noqa: F811
    """Live loop on the final board (stuck times shortened): one tap, flagged ``lost``. The
    arrow frame reads the belt's labels (yellow 27 %, darkred 8 % / 57 %; 00238 reads none of
    red / yellow, and a colour never read has no stall clock)."""
    import dataclasses

    from analyzer import Analyzer
    from runlog import load_events
    from tests.test_bot import Tapper, make_bot, run

    # gamble_fill_margin 100: no bucket can fill, so the last-slot gamble finds none (the arrow
    # frame does not show the stalled red on the belt: the gamble would take r2c2 itself)
    sc = dataclasses.replace(cfg.strategy, stall_s=0.2, gamble_stuck_s=0.2, lost_stuck_s=0.5,
                             gamble_fill_margin=100.0)
    c = dataclasses.replace(cfg, strategy=sc)
    tapper = Tapper()
    bot, log = make_bot(c, Analyzer(c, clock=lambda: 0.0), tmp_path, [frame("unknown-092231-354")] * 40,
                        tapper=tapper, max_taps=1)
    run(bot, log)
    taps = [e for e in load_events(log.dir) if e["type"] == "tap"]
    assert len(taps) == 1 and taps[0]["pos"] == [2, 2] and taps[0]["lost"] is True
