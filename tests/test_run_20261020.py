"""Dry run 20261020-120601 (L187 HARD start, shot 064): the 2x column pitch, new colours yellow +
purple, yellow + blue locks / keys. The shot is annotated in dataset/annotations.json, so
tests/test_vision.py also checks its lattice and every cell; these are the run's failures."""
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
