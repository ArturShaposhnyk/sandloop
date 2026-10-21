"""Pure logic: path rule (BFS), cell notation, lock registry. No screenshots needed."""
import pytest

pytest.importorskip("cv2")

from cells import CELL_LOCKS, Cell, CellFeatures, Kind, classify_cell, path_playable
from config import VisionConfig
from palette import Palette


def grid_from(rows):
    """'B' bucket, '.' empty, 'L' lock, 'x' no cell."""
    out = []
    for r, row in enumerate(rows):
        line = []
        for c, ch in enumerate(row.split()):
            kind = {"B": Kind.BUCKET, ".": Kind.EMPTY, "L": Kind.LOCK}.get(ch)
            line.append(None if ch == "x" else Cell(r, c, kind))
        out.append(line)
    return out


def exits_of(grid):
    return [(0, c) for c, cell in enumerate(grid[0]) if cell is not None]


def playable_set(rows):
    g = grid_from(rows)
    p = path_playable(g, exits_of(g))
    return {(r, c) for r, row in enumerate(p) for c, ok in enumerate(row) if ok}


def test_front_row_buckets_are_playable():
    assert playable_set(["B B", "B B"]) == {(0, 0), (0, 1)}


def test_side_path_through_empty_cells():
    # bucket at (2, 1) is reached via the empty column 0, not from directly in front
    assert (2, 1) in playable_set([". B B",
                                   ". B B",
                                   ". B B"])


def test_enclosed_bucket_is_closed():
    assert (1, 1) not in playable_set(["B B B",
                                       "B B B",
                                       "B . B"])   # the empty cell below is itself enclosed


def test_locks_block_the_path():
    assert playable_set(["L B", ". B"]) == {(0, 1)}   # (1, 0) is empty but behind the lock


def test_cross_shaped_grid_exit_only_through_the_neck():
    rows = ["x x . x x",
            "x x . x x",
            "B B . . B",
            "B B B . B",
            "x B B B x"]
    p = playable_set(rows)
    assert (2, 1) in p and (3, 2) in p and (4, 3) in p and (2, 4) in p and (3, 4) in p
    assert (2, 0) not in p and (3, 0) not in p and (4, 1) not in p and (4, 2) not in p


def test_cell_label_notation():
    assert Cell(0, 0, Kind.EMPTY).label() == "."
    assert Cell(0, 0, Kind.LOCK, lock="ice", counter=17).label() == "ice17"
    assert Cell(0, 0, Kind.LOCK, lock="curtain", counter=3).label() == "cur3"
    assert Cell(0, 0, Kind.BUCKET, color="red", face_up=True).label() == "red-up"
    assert Cell(0, 0, Kind.BUCKET, color="cream", body="red", face_up=False).label() == "cream/red-down"
    assert Cell(0, 0, Kind.UNKNOWN).label() == "?"
    assert Cell(0, 0, Kind.UNKNOWN).is_obstacle and not Cell(0, 0, Kind.EMPTY).is_obstacle


def test_lock_registry_is_data():
    names = [lock.name for lock in CELL_LOCKS]
    assert len(names) == len(set(names)) and "ice" in names
    assert all(lock.unlock in ("taps", "fills", "key", "path", "adjacent", "drops")
               for lock in CELL_LOCKS)
    assert {"lock", "key", "chest", "mystery"} <= set(names)
    assert all(lock.counter in ("optional", "required", "none") for lock in CELL_LOCKS)


def test_chest_and_mystery_from_colour_shares():
    from config import VisionConfig

    v = VisionConfig()
    chest = classify_cell(0, 0, feats(wood_frac=0.33), PAL, v)
    assert chest.kind is Kind.LOCK and chest.lock == "chest" and chest.unlock == "path"
    assert Cell(0, 0, Kind.LOCK, lock="chest", counter=4).label() == "chest4"
    mystery = classify_cell(0, 0, feats(grey_frac=0.5), PAL, v)
    assert mystery.kind is Kind.LOCK and mystery.label() == "mystery"
    assert classify_cell(0, 0, feats(wood_frac=0.06, grey_frac=0.1), PAL, v).kind is Kind.BUCKET


PAL = Palette(buckets={"red": [(50.0, 60.0, 40.0)], "blue": [(35.0, 20.0, -60.0)]},
              ui={"empty_cell": (90.0, -2.0, -10.0), "panel_bg": (62.0, 5.0, -20.0)})


def feats(**kw):
    base = dict(empty_frac=0.0, white_frac=0.0, dark_frac=0.0, face_ratio=1.0,
                top_lab=(50.0, 60.0, 40.0), body_lab=(50.0, 60.0, 40.0), hue_cyan_frac=0.0)
    base.update(kw)
    return CellFeatures(**base)


def test_classify_empty_ice_bucket_unknown():
    v = VisionConfig()
    assert classify_cell(0, 0, feats(empty_frac=0.8), PAL, v).kind is Kind.EMPTY
    ice = classify_cell(0, 0, feats(white_frac=0.2, dark_frac=0.08, hue_cyan_frac=0.5), PAL, v)
    assert ice.kind is Kind.LOCK and ice.lock == "ice" and ice.unlock == "taps"
    up = classify_cell(0, 0, feats(face_ratio=0.8), PAL, v)
    assert up.kind is Kind.BUCKET and up.color == "red" and up.face_up and up.body is None
    unk = classify_cell(0, 0, feats(top_lab=(70.0, -60.0, 70.0), body_lab=(70.0, -60.0, 70.0)), PAL, v)
    assert unk.kind is Kind.UNKNOWN and "palette" in unk.reason


def test_tower_needs_a_clearly_different_second_colour():
    v = VisionConfig()
    tower = classify_cell(0, 0, feats(body_lab=(35.0, 20.0, -60.0)), PAL, v)
    assert tower.color == "red" and tower.body == "blue"
