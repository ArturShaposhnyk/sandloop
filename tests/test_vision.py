"""Vision on the saved, annotated screenshots (dataset/annotations.json is the ground truth)."""
import json
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

from cells import Cell, Kind, path_playable
from config import load_config
from digits import DigitReader
from palette import Palette
from tools.build_palette import grid_labels, parse_label, read_shot
from icons import IconMatcher
from unknown_colours import save_unknown
from vision import find_lattice, read_gameplay, read_grid

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
ANN = json.loads((DATASET / "annotations.json").read_text())
GRIDS = grid_labels(ANN)

# Cells the current vision gets wrong, with the reason (kept visible, not hidden).
KNOWN_LIMITATIONS: dict = {}


@pytest.fixture(scope="module")
def ctx():
    cfg = load_config(ROOT / "config.yaml")
    return (cfg.vision, Palette.load(DATASET / "palette.json"),
            DigitReader.load(DATASET / "digits.npz"),
            IconMatcher.load(DATASET / "templates" / "icons"))


def view(seq, ctx):
    v, pal, rd, icons = ctx
    return read_gameplay(read_shot(DATASET, seq), pal, v, rd, icons=icons, log=lambda s: None)


@pytest.mark.parametrize("seq", sorted(GRIDS))
def test_lattice_shape_matches_annotation(seq, ctx):
    v, pal, _, _ = ctx
    lat = find_lattice(read_shot(DATASET, seq), pal, v)
    rows = GRIDS[seq]
    assert (lat.rows, lat.cols) == (len(rows), len(rows[0]))
    assert [[bool(m) for m in r] for r in lat.mask] == [[l != "x" for l in r] for r in rows]


@pytest.mark.parametrize("seq", sorted(GRIDS))
def test_grid_cells_match_annotation(seq, ctx):
    v, pal, rd, icons = ctx
    _, grid, _ = read_grid(read_shot(DATASET, seq), pal, v, rd, icons)
    wrong = []
    for r, row in enumerate(GRIDS[seq]):
        for c, want in enumerate(row):
            got = "x" if grid[r][c] is None else grid[r][c].label()
            if got != want and (seq, r, c) not in KNOWN_LIMITATIONS:
                wrong.append(f"r{r}c{c}: want {want}, got {got}")
    assert not wrong, wrong


def _annotated_cells(rows):
    grid = []
    for r, row in enumerate(rows):
        line = []
        for c, label in enumerate(row):
            kind, color, body, up, _ = parse_label(label)
            k = {"empty": Kind.EMPTY, "bucket": Kind.BUCKET, "none": None}.get(kind, Kind.LOCK)
            line.append(None if k is None else Cell(r, c, k, color=color, face_up=up))
        grid.append(line)
    return grid


@pytest.mark.parametrize("seq", sorted(GRIDS))
def test_path_rule_agrees_with_annotated_faces(seq):
    """docs/GAME_RULES.md path rule vs. what the screenshots show (hand labels)."""
    grid = _annotated_cells(GRIDS[seq])
    exits = [(0, c) for c, cell in enumerate(grid[0]) if cell is not None]
    play = path_playable(grid, exits)
    expected = {tuple(x) for x in ANN["visual_vs_path_disagreements"].get(seq, [])}
    got = {(r, c) for r, row in enumerate(grid) for c, cell in enumerate(row)
           if cell is not None and cell.kind is Kind.BUCKET and cell.face_up != play[r][c]}
    assert got == expected


def test_disagreement_is_logged_and_not_tappable(ctx):
    v, pal, rd, icons = ctx
    logs = []
    gv = read_gameplay(read_shot(DATASET, "010"), pal, v, rd, icons=icons, log=logs.append)
    assert (4, 4) in gv.disagreements
    assert (4, 4) not in gv.tappable()
    assert any("r4c4" in line and "not tapping" in line for line in logs)
    assert set(gv.tappable()) == {(2, 2), (3, 2), (3, 4)}


@pytest.mark.parametrize("seq,counter", [("001", 12), ("004", 10), ("008", 1)])
def test_curtain_extent_and_counter(seq, counter, ctx):
    gv = view(seq, ctx)
    assert len(gv.regions) == 1
    reg = gv.regions[0]
    assert sorted(reg.cells) == [(2, 2), (2, 3), (3, 2), (3, 3)]
    assert reg.lock == "curtain" and reg.unlock == "fills" and reg.counter == counter


@pytest.mark.parametrize("seq", sorted(ANN["counters"]))
def test_conveyor_counter(seq, ctx):
    assert view(seq, ctx).counter == int(ANN["counters"][seq].split("/")[0])


@pytest.mark.parametrize("seq", sorted(ANN["fill_percent"]))
def test_fill_percent_labels(seq, ctx):
    gv = view(seq, ctx)
    assert [s.fill for s in gv.slots if s.fill is not None] == ANN["fill_percent"][seq]
    assert all(s.color is not None for s in gv.slots if s.fill is not None)


def test_sand_full_picture_and_exposed_bottom(ctx):
    gv = view("001", ctx)
    level_colors = {"blue", "pink", "magenta", "cream", "cyan"}
    assert set(gv.sand) == level_colors
    assert abs(sum(gv.sand.values()) - 1.0) < 0.05
    assert max(gv.exposed, key=gv.exposed.get) == "blue"   # the picture's bottom stripe


def test_tap_points_are_inside_the_screen(ctx):
    gv = view("043", ctx)
    for r, c in gv.tappable():
        x, y = gv.cell_center_rel(r, c)
        assert 0 < x < 1 and 0.5 < y < 0.9


def test_save_unknown_writes_frame_crop_and_log(ctx, tmp_path):
    v, pal, rd, icons = ctx
    img = read_shot(DATASET, "043")
    gv = read_gameplay(img, pal, v, rd, icons=icons, log=lambda s: None)
    gv.grid[3][0] = Cell(3, 0, Kind.UNKNOWN, reason="test obstacle")
    logs = []
    saved = save_unknown(img, gv, tmp_path, log=logs.append)
    assert len(saved) == 2 and all(p.exists() for p in saved)
    rec = json.loads((tmp_path / "unknown.jsonl").read_text().strip())
    assert rec["row"] == 3 and rec["col"] == 0 and rec["reason"] == "test obstacle"
    assert logs and "r3c0" in logs[0]
    assert save_unknown(img, read_gameplay(img, pal, v, rd, icons=icons, log=lambda s: None), tmp_path) == []


# Live crops of the iPhone screen can be off by a few pixels (pt/px rounding, the window's
# anti-aliased edge, the hover-chrome frame). A 3 px dark strip used to make every grid read
# fail ("grid panel background not found") and a 4-10 px vertical offset lost the k/5 counter.
def _misalign(img, dx=0, dy=0, colour=(25, 26, 27)):
    """Shift the content by dx/dy pixels (same frame size); the uncovered strip is ``colour``."""
    import cv2

    t, b, l, r = max(dy, 0), max(-dy, 0), max(dx, 0), max(-dx, 0)
    out = cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=colour)
    h, w = img.shape[:2]
    return out[b:b + h, r:r + w]


MISALIGNMENTS = [(3, 0), (8, 0), (-3, 0), (-8, 0), (0, 4), (0, -4), (0, 10), (0, -10), (6, -6)]


@pytest.mark.parametrize("dx,dy", MISALIGNMENTS)
@pytest.mark.parametrize("seq", sorted(GRIDS))
def test_grid_survives_misaligned_crop(seq, dx, dy, ctx):
    v, pal, rd, icons = ctx
    _, grid, _ = read_grid(_misalign(read_shot(DATASET, seq), dx, dy), pal, v, rd, icons)
    rows = GRIDS[seq]
    assert (len(grid), len(grid[0])) == (len(rows), len(rows[0]))
    got = [["x" if c is None else c.label() for c in row] for row in grid]
    wrong = [f"r{r}c{c}: want {want}, got {got[r][c]}" for r, row in enumerate(rows)
             for c, want in enumerate(row)
             if got[r][c] != want and (seq, r, c) not in KNOWN_LIMITATIONS]
    assert not wrong, wrong


@pytest.mark.parametrize("dx,dy", MISALIGNMENTS)
@pytest.mark.parametrize("seq", sorted(ANN["counters"]))
def test_counter_survives_misaligned_crop(seq, dx, dy, ctx):
    v, pal, rd, icons = ctx
    gv = read_gameplay(_misalign(read_shot(DATASET, seq), dx, dy), pal, v, rd, icons=icons,
                       log=lambda s: None)
    assert gv.counter == int(ANN["counters"][seq].split("/")[0])


def test_l183_three_curtains_locks_and_keys(ctx):
    gv = view("053", ctx)
    regs = sorted((sorted(r.cells), r.counter) for r in gv.regions)
    assert regs == [([(1, 3), (1, 4), (2, 3), (2, 4)], 10),
                    ([(5, 1), (5, 2), (6, 1), (6, 2)], 15),
                    ([(5, 4), (5, 5), (6, 4), (6, 5)], 20)]
    icons = {(c.row, c.col): (c.lock, c.unlock, c.color) for row in gv.grid for c in row
             if c is not None and c.lock in ("lock", "key")}
    assert icons == {(2, 1): ("lock", "key", "yellow"), (3, 5): ("lock", "key", "red"),
                     (2, 6): ("key", "path", "yellow"), (4, 2): ("key", "path", "red")}
    assert set(gv.tappable()) == {(0, 2), (0, 3), (0, 4), (0, 5)}
