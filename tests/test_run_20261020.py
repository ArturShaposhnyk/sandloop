"""Dry run 20261020-120601 (L187 HARD start, shot 064): the 2x column pitch, new colours yellow +
purple, yellow + blue locks / keys. The shot is annotated in dataset/annotations.json, so
tests/test_vision.py also checks its lattice and every cell; these are the run's failures."""
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from state import LevelState
from strategy import choose
from tests.test_state import analyzer, shot  # noqa: F401  (fixture)
from vision import _fundamental, find_lattice, read_gameplay


@pytest.fixture(scope="module")
def view(analyzer):  # noqa: F811
    return read_gameplay(shot("064"), analyzer.palette, analyzer.cfg.vision, analyzer.reader,
                         log=lambda s: None, icons=analyzer.icons, hints=analyzer.hints)


def test_lattice_is_7x7_not_the_2x_pitch(analyzer):  # noqa: F811
    """Walls every two columns made 112 px (2x) the strongest autocorrelation peak: the run read
    a 4x4 lattice and every cell spanned several buckets ("cream-up looks face-up" warnings)."""
    lat = find_lattice(shot("064"), analyzer.palette, analyzer.cfg.vision)
    assert (lat.rows, lat.cols) == (7, 7) and 54 < lat.cw < 59


def test_fundamental_prefers_a_strong_half_pitch():
    ac = np.zeros(200)
    ac[0], ac[56], ac[112] = 1.0, 0.50, 0.51
    assert _fundamental(ac, 112, 35, 0.8) == 56
    ac[56] = 0.3                                  # a weak half-pitch peak is not the pitch
    assert _fundamental(ac, 112, 35, 0.8) == 112


def test_new_colours_locks_and_keys(view):
    labels = [row.split() for row in view.labels()]
    assert labels[1][2] == labels[2][2] == "yellow-down"
    assert [labels[r][c] for r, c in ((3, 0), (4, 4), (5, 3))] == ["purple-down"] * 3
    assert labels[1][5] == "purple/cream-down"            # purple disc on a cream dome
    assert (labels[2][0], labels[2][6]) == ("lock:yellow", "lock:blue")
    assert (labels[6][2], labels[6][4]) == ("key:yellow", "key:blue")   # was orange-down / ?
    assert labels[6][0] == labels[6][6] == "."            # empty, not keys (icon crop fix)
    assert view.unknown_colours == []


def test_new_sand_colours(view):
    assert 0.25 < view.sand["purple"] < 0.33              # the sky
    assert 0.06 < view.sand["yellow"] < 0.11


def test_start_decision(view, analyzer):  # noqa: F811
    st = LevelState.from_view(view, analyzer.cfg.strategy.conveyor_slots,
                              analyzer.cfg.strategy.unknown_fill_need)
    assert set(st.tappable) == {(0, 2), (0, 3), (0, 4)}
    d = choose(st, analyzer.cfg.strategy)
    assert (d.move.pos, d.move.color) == ((0, 3), "orange")


# --- live run 20261020-122638 (L187, STUCK at 3/5): the lattice turned 4x4 mid-level ---------

RUN = Path(__file__).resolve().parent / "data" / "run-20261020-122638"
STUCK = ["x x . . darkred-up x x",
         "orange/cream-up . . . purple-up purple/cream-down cream-down",
         "lock:yellow x yellow-up . . x lock:blue",
         "purple-down x darkred-up . . x purple-down",
         "orange-down x blue-up . purple-up x orange-down",
         "purple-down x orange-down purple-up orange-down x purple-down",
         ". x key:yellow blue-down key:blue x ."]


def frame(name):
    img = cv2.imread(str(RUN / f"{name}.png"))
    assert img is not None, name
    return img


def read_frame(analyzer, name):  # noqa: F811
    return read_gameplay(frame(name), analyzer.palette, analyzer.cfg.vision, analyzer.reader,
                         log=lambda s: None, icons=analyzer.icons, hints=analyzer.hints)


@pytest.mark.parametrize("name", ["00030-wait", "00041-stuck-disagree"])
def test_emptied_lanes_keep_the_7x7_lattice(analyzer, name):  # noqa: F811
    """As the middle lanes emptied the half-pitch peak faded (0.38 vs 0.52 at 2x) and the run
    read 4x4. The lane walls sit on 56.6 px boundaries only: wall alignment keeps 7x7."""
    gv = read_frame(analyzer, name)
    assert (gv.lattice.rows, gv.lattice.cols) == (7, 7) and 54 < gv.lattice.cw < 59
    assert gv.labels() == STUCK
    assert gv.unknown_colours == []                     # LAB 74/19/0 was a merged 4x4 cell


def test_stuck_frame_has_moves(analyzer):  # noqa: F811
    gv = read_frame(analyzer, "00041-stuck-disagree")
    assert not gv.disagreements
    assert {(0, 4), (1, 4), (2, 2), (3, 2), (4, 2), (4, 4), (5, 3)} <= set(gv.tappable())


def test_open_tower_keeps_its_lower_stage(analyzer):  # noqa: F811
    """Open tower in a tall lane: the cream stage is drawn below the bucket crop (it used to
    read orange-up); vision.foot_strip sees it."""
    gv = read_frame(analyzer, "00041-stuck-disagree")
    assert gv.labels()[1].split()[0] == "orange/cream-up"   # orange/cream-down at the start


def test_lattice_lock_never_takes_a_coarser_lattice_with_the_same_outline():
    from hints import LatticeLock
    from vision import Lattice

    lock = LatticeLock(lock_frames=1)
    fine = Lattice(101.0, 748.0, 56.6, 50.1, np.ones((7, 7), bool))
    coarse = Lattice(101.0, 748.0, 99.0, 87.6, np.ones((4, 4), bool))   # the run's misread
    assert lock.check(fine, log=lambda s: None) is fine
    for _ in range(10):                                  # held any number of frames: not used
        assert lock.check(coarse, log=lambda s: None) is fine
    assert lock.check(fine, log=lambda s: None) is fine


def test_two_buckets_in_one_cell_are_never_a_tower(analyzer):  # noqa: F811
    """The run's 4x4 cells held two buckets of a lane ("purple/cream-up" r3c0, "purple/cream-up"
    r3c3): a background row through the middle with objects above and below = two objects ->
    UNKNOWN, never tapped. (Touching pairs such as r2c1 "cream/blue-up" score ~0.1 and are
    not caught: the lattice checks above are the main guard.)"""
    from cells import Kind, cell_features, classify_cell
    from vision import Lattice, cell_crop

    v = analyzer.cfg.vision
    img = frame("00041-stuck-disagree")
    coarse = Lattice(101.0, 748.0, 99.0, 87.6, np.ones((4, 4), bool))
    for r, c in ((3, 0), (3, 3), (3, 1), (2, 3)):
        f = cell_features(cell_crop(img, coarse, r, c, v), analyzer.palette, v)
        cell = classify_cell(r, c, f, analyzer.palette, v)
        assert cell.kind is Kind.UNKNOWN and "two objects" in cell.reason, (r, c, cell.label())


def test_one_object_filling_part_of_a_cell_is_not_two(analyzer):  # noqa: F811
    """Fading padlocks right after a lock opened (run 20261017) fill only the top of the cell."""
    from cells import cell_features
    from vision import cell_crop

    v = analyzer.cfg.vision
    run17 = RUN.parent / "run-20261017-103804"
    for name, (r, c) in (("00285-wait", (2, 1)), ("00304-tap", (3, 5))):
        img = cv2.imread(str(run17 / f"{name}.png"))
        lat = find_lattice(img, analyzer.palette, v)
        f = cell_features(cell_crop(img, lat, r, c, v), analyzer.palette, v)
        assert f.split_bg < v.split_bg_max / 2


# --- live run 20261020-125351 (L187): 21 good taps, then the lattice flip-flopped 7x7 <-> 4x4 --

RUN2 = Path(__file__).resolve().parent / "data" / "run-20261020-125351"
END = ["x x . . darkred-up x x",
       ". . . . . . cream-up",
       ". x yellow-up . cream-up x .",
       ". x darkred-up . . x purple-down",
       ". x . . . x orange-down",
       ". x . . . x purple-down"]
# hint arrows over the boosters (partly or not masked) / sparkles joined to the grid region
MISREAD = ["00047-wait"]
# ... the same, but since run 20261021-092045 the 4 arrows are masked as a row (hints._in_row);
# 00147 / 125540-121 since run 20261021-143729: arrows searched below grid_zone
# (vision.hint_zone_bottom) - they had bobbed out of the search zone
ROW_MASKED = ["00143-wait", "unknown-125531-186", "00147-wait", "unknown-125540-121"]


def frame2(name):
    img = cv2.imread(str(RUN2 / f"{name}.png"))
    assert img is not None, name
    return img


def read2(analyzer, name, lattices=None):  # noqa: F811
    return read_gameplay(frame2(name), analyzer.palette, analyzer.cfg.vision, analyzer.reader,
                         log=lambda s: None, icons=analyzer.icons, hints=analyzer.hints,
                         lattices=lattices)


def lock(analyzer):  # noqa: F811
    from hints import LatticeLock

    v = analyzer.cfg.vision
    return LatticeLock(v.lattice_lock_frames, v.lattice_min_fit)


@pytest.mark.parametrize("name", MISREAD)
def test_grid_joined_by_arrows_or_sparkles_is_a_misfit(analyzer, name):  # noqa: F811
    """The grown outline (68..531 px instead of 101..497) reads 4x4 and its lane walls sit off
    the cell boundaries; clean frames fit >= 0.88."""
    gv = read2(analyzer, name)
    assert gv.lattice.cols == 4 and gv.lattice.fit < analyzer.cfg.vision.lattice_min_fit


@pytest.mark.parametrize("name", ROW_MASKED)
def test_arrow_row_masked_reads_7x7_without_a_lock(analyzer, name):  # noqa: F811
    """The 4 booster arrows match 0.6-0.8 each; as a row (same height and scale) all are masked
    and the frame reads the level's 7x7 by itself."""
    gv = read2(analyzer, name)
    assert (gv.lattice.rows, gv.lattice.cols) == (7, 7)
    assert gv.labels()[:6] == END


@pytest.mark.parametrize("name", ["00002-tap", "00137-wait"])
def test_clean_frames_fit_7x7(analyzer, name):  # noqa: F811
    gv = read2(analyzer, name)
    assert (gv.lattice.rows, gv.lattice.cols) == (7, 7)
    assert gv.lattice.fit >= analyzer.cfg.vision.lattice_min_fit


def test_misfit_frame_never_becomes_the_first_lock(analyzer):  # noqa: F811
    """The run's first lock was 7x7 (frame 00002); a level entered while the arrows show must
    not lock 4x4 either: a misfit reading is a failed read until a fitting one comes."""
    from vision import VisionError

    lk = lock(analyzer)
    with pytest.raises(VisionError, match="walls off"):
        read2(analyzer, "00047-wait", lk)
    assert not lk.locked
    gv = read2(analyzer, "00137-wait", lk)
    assert lk.locked and (gv.lattice.rows, gv.lattice.cols) == (7, 7)


def test_locked_lattice_reads_the_flip_flop_frames(analyzer):  # noqa: F811
    """After the lock every frame is read with the level's 7x7, whatever it reads itself: no
    "new grid geometry", no merged "two objects" cells, no orange/cream-down r1c1 warning."""
    lk = lock(analyzer)
    assert read2(analyzer, "00137-wait", lk).labels()[:6] == END
    for name in (MISREAD[1:] + ROW_MASKED) * 2:      # held any number of frames (00047: earlier board)
        gv = read2(analyzer, name, lk)
        assert (gv.lattice.rows, gv.lattice.cols) == (7, 7), name
        assert gv.labels()[:6] == END, name
        assert not gv.disagreements, name
        assert all(c.row == 6 for c in gv.unknown_cells()), name   # at most an arrow-hidden cell
