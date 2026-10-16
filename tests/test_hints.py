"""Game hint arrows (hints.py): live regression 055 (L183, 4/5 on the belt, yellow arrows over
the bottom row pointing at the boosters)."""
import dataclasses
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

from cells import Kind  # noqa: E402
from config import load_config  # noqa: E402
from digits import DigitReader  # noqa: E402
from hints import CellMemory, HintArrows  # noqa: E402
from icons import IconMatcher  # noqa: E402
from palette import Palette  # noqa: E402
from strategy import choose  # noqa: E402
from state import LevelState  # noqa: E402
from tools.build_palette import read_shot  # noqa: E402
from vision import find_lattice, read_gameplay  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
SCREENS = sorted((DATASET / "screens").glob("*.png"))

GRID_055 = [
    "x x . . . . x",
    "x . . cur4 cur4 . x",
    "x . x cur4 cur4 . .",
    "cream-up . x x x . x",
    "green-up . . x green-down cream-up green/cyan-down",   # r4c0: open LIGHT green (user)
    "darkgreen-down cur9 cur9 x cur14 cur14 darkgreen-down",
    "cyan-down cur9 cur9 x cur14 cur14 orange-down",
]


@pytest.fixture(scope="module")
def ctx():
    cfg = load_config(ROOT / "config.yaml")
    return (cfg, Palette.load(DATASET / "palette.json"), DigitReader.load(DATASET / "digits.npz"),
            IconMatcher.load(DATASET / "templates" / "icons"),
            HintArrows.load(DATASET / "templates" / "hints"))


def view(seq, ctx, v=None, memory=None):
    cfg, pal, rd, icons, hints = ctx
    return read_gameplay(read_shot(DATASET, seq), pal, v or cfg.vision, rd, icons=icons,
                         hints=hints, memory=memory, log=lambda s: None)


def test_arrows_found_only_on_055(ctx):
    cfg, _, _, _, hints = ctx
    for path in SCREENS:
        boxes = [b for b, _ in hints.find(cv2.imread(str(path)), cfg.vision)]
        seq = path.stem.split("-")[3]
        if seq == "055":
            assert sorted(b[0] for b in boxes) == pytest.approx([72, 202, 332, 462], abs=4)
            assert all(abs(b[1] - 1074) <= 4 for b in boxes)
        else:
            assert boxes == [], seq


def test_055_lattice_is_not_joined_to_the_boosters(ctx):
    """Without the mask the arrows connect the grid to the booster buttons: 8 rows, 8 columns,
    a phantom row of "orange buckets". With it the lattice is the one of 054 (same level)."""
    cfg, pal, _, _, _ = ctx
    gv = view("055", ctx)
    lat054 = find_lattice(read_shot(DATASET, "054"), pal, cfg.vision)
    assert (gv.lattice.rows, gv.lattice.cols) == (7, 7)
    assert (gv.lattice.x0, gv.lattice.y0, gv.lattice.cw) == pytest.approx(
        (lat054.x0, lat054.y0, lat054.cw), abs=1)
    assert len(gv.hints) == 4


def test_055_cells_under_arrows_read_through_the_mask(ctx):
    gv = view("055", ctx)
    assert gv.labels() == GRID_055
    assert gv.unknown_cells() == [] and gv.unknown_colours == []
    assert set(gv.tappable()) == {(3, 0), (4, 0), (4, 5)}


def test_hidden_cell_reuses_last_clean_reading(ctx):
    """A cell too hidden to read gets the reading of an earlier clean frame of the same
    lattice (054: cyan r6c0, orange r6c6); it is never tapped nor saved as unknown."""
    cfg = ctx[0]
    strict = dataclasses.replace(cfg.vision, hint_cover_max=0.1)
    mem = CellMemory()
    view("054", ctx, memory=mem)
    gv = view("055", ctx, v=strict, memory=mem)
    left, right = gv.grid[6][0], gv.grid[6][6]
    assert left.occluded and left.label() == "cyan-down"
    assert right.occluded and right.label() == "orange-down"
    assert gv.unknown_cells() == []


def test_hidden_cell_without_memory_is_unknown_but_not_a_new_mechanic(ctx):
    cfg = ctx[0]
    strict = dataclasses.replace(cfg.vision, hint_cover_max=0.1)
    gv = view("055", ctx, v=strict)
    cell = gv.grid[6][0]
    assert cell.kind is Kind.UNKNOWN and cell.occluded and "hint arrow" in cell.reason
    assert gv.unknown_cells() == []       # not saved to dataset/unknown/


def test_055_taps_the_light_green_bucket(ctx):
    """User: r4c0 is an open LIGHT green bucket (its shaded inside was read darkgreen, 24.6 %
    exposed). Green is 47 % of the contact layer, so even at 4/5 (last slot: avail >= 25 %)
    it is the move; the creams have no sand at the bottom (reason on every candidate)."""
    cfg = ctx[0]
    st = LevelState.from_view(view("055", ctx), cfg.strategy.conveyor_slots,
                              cfg.strategy.unknown_fill_need)
    d = choose(st, cfg.strategy)
    assert d.move is not None and (d.move.pos, d.move.color) == ((4, 0), "green")
    reasons = {c.pos: c.rejected for c in d.candidates}
    assert set(reasons) == {(3, 0), (4, 0), (4, 5)}
    assert reasons[(4, 0)] is None
    assert "< 25%, last slot" in reasons[(3, 0)] and "< 25%, last slot" in reasons[(4, 5)]


@pytest.mark.parametrize("seq", ["053", "054", "055"])
def test_l183_greens_are_not_close_calls(seq, ctx):
    """Every green / darkgreen bucket on the L183 shots is reported with its distance to both
    greens, and none of them is a close call (open buckets are read from the rim)."""
    gv = view(seq, ctx)
    greens = [(c.row, c.col) for row in gv.grid for c in row
              if c is not None and c.kind is Kind.BUCKET and c.color in ("green", "darkgreen")]
    notes = [n for n in gv.colour_notes if "green" in n]
    assert len(notes) == len(greens) and greens
    assert not any("CLOSE" in n for n in notes)
