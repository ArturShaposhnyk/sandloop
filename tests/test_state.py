"""state.py on synthetic grids (pure logic) and on saved screenshots."""
from typing import Dict, List, Optional

import pytest

from cells import Cell, Kind
from state import ConveyorBucket, LevelState, Lock


def make_state(rows: List[str], k: Optional[int] = 0, conveyor=(), sand: Optional[Dict] = None,
               exposed: Optional[Dict] = None, locks=()) -> LevelState:
    """Rows of space-separated cells: '.' empty, 'x' no cell, 'ice3' lock, 'cur2' curtain cell,
    'red' face-up bucket, 'red_' face-down bucket, 'red/blue' tower (top/body)."""
    grid = []
    for r, line in enumerate(rows):
        out = []
        for c, t in enumerate(line.split()):
            if t == "x":
                out.append(None)
            elif t == ".":
                out.append(Cell(r, c, Kind.EMPTY))
            elif t.startswith("ice"):
                out.append(Cell(r, c, Kind.LOCK, lock="ice", unlock="taps", counter=int(t[3:])))
            elif t.startswith("cur"):
                out.append(Cell(r, c, Kind.LOCK, lock="curtain", unlock="fills",
                                counter=int(t[3:]), region=0))
            else:
                top, _, body = t.rstrip("_").partition("/")
                out.append(Cell(r, c, Kind.BUCKET, color=top, body=body or None,
                                face_up=not t.endswith("_")))
        grid.append(out)
    exits = [(0, c) for c, cell in enumerate(grid[0]) if cell is not None]
    locks = list(locks)
    for row in grid:
        for cell in row:
            if cell is not None and cell.kind is Kind.LOCK and cell.region is None:
                locks.append(Lock(cell.lock, cell.unlock, cell.counter, ((cell.row, cell.col),)))
    st = LevelState(grid, exits, [], [], locks, k, 5, list(conveyor), sand or {}, exposed or {})
    st.tappable = st.playable()
    return st


def test_playable_follows_the_path_rule():
    st = make_state(["red . blue",
                     "pink green_ cyan_",
                     "ice2 . red_"])
    assert set(st.playable()) == {(0, 0), (0, 2), (1, 1)}   # pink r1c0 is boxed in


def test_after_tap_empties_cell_counts_down_ice_and_fills_a_slot():
    st = make_state(["red blue",
                     "ice2 pink_"], k=1, conveyor=[ConveyorBucket("cyan", 40)])
    nxt = st.after_tap((0, 1))
    assert nxt.cell((0, 1)).kind is Kind.EMPTY
    assert nxt.conveyor_k == 2 and nxt.conveyor[-1] == ConveyorBucket("blue", 0)
    assert [l.counter for l in nxt.locks] == [1] and nxt.cell((1, 0)).counter == 1
    assert (1, 1) in nxt.tappable                 # pink is now connected to the exit
    assert st.cell((0, 1)).kind is Kind.BUCKET    # original state untouched
    assert st.locks[0].counter == 2


def test_tower_tap_keeps_the_next_stage_as_an_obstacle():
    st = make_state(["red/blue",
                     "pink_"])
    nxt = st.after_tap((0, 0))
    cell = nxt.cell((0, 0))
    assert cell.kind is Kind.LOCK and cell.lock == "tower" and cell.color == "blue"
    assert nxt.tappable == []
    assert st.opened_by((0, 0)) == ([], [])


def test_opened_by_reports_buckets_and_lock_cells():
    st = make_state([". red .",
                     "ice3 blue_ green_",
                     "x pink_ x"])
    buckets, locks = st.opened_by((0, 1))
    assert buckets == [(1, 1)] and locks == []
    st2 = make_state(["red . blue",
                      "ice3 green_ pink_"])
    assert st2.opened_by((0, 0)) == ([], [(1, 0)])


def test_after_tap_rejects_non_buckets():
    st = make_state([". ice1"])
    with pytest.raises(ValueError):
        st.after_tap((0, 0))


def test_bucket_counts_hidden_and_capacity_estimate():
    cur = Lock("curtain", "fills", 3, ((1, 0), (1, 1)))
    st = make_state(["red red/blue cyan_",
                     "cur3 cur3 ice2"], k=2,
                    conveyor=[ConveyorBucket("red", 50), ConveyorBucket("cyan", None)],
                    sand={"red": 0.3, "blue": 0.1, "cyan": 0.2}, locks=[cur])
    assert st.grid_buckets() == {"red": 2, "blue": 1, "cyan": 1}
    assert st.hidden_buckets == 2
    assert st.conveyor_need("red") == 0.5 and st.conveyor_need("cyan") == 0.5
    assert st.pending_buckets() == 4 + 2 + 1.0       # ice is not counted
    assert st.capacity_estimate() == pytest.approx(0.6 / 7)
    assert st.free_slots == 3


def test_free_slots_unknown_without_counter():
    assert make_state(["red"], k=None).free_slots is None
    assert make_state(["red"], k=5).free_slots == 0


# --- on saved screenshots --------------------------------------------------------------------

@pytest.fixture(scope="module")
def analyzer():
    pytest.importorskip("cv2")
    from pathlib import Path

    from config import load_config
    from strategy import Analyzer

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config.yaml")
    cfg.paths.palette = str(root / cfg.paths.palette)
    cfg.paths.dataset = str(root / cfg.paths.dataset)
    return Analyzer(cfg)


def shot(seq):
    import cv2 as _cv2
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "dataset" / "screens"
    return _cv2.imread(str(next(root.glob(f"*-{seq}-auto.png"))))


def test_state_from_view_l180_start(analyzer):
    st = analyzer.analyze(shot("001"), log=lambda s: None).state
    assert st.conveyor_k == 0 and st.conveyor == [] and st.free_slots == 5
    assert st.grid_buckets() == {"blue": 4, "pink": 4, "magenta": 3, "cream": 2, "cyan": 2}
    assert st.hidden_buckets == 4
    kinds = sorted((l.kind, l.counter) for l in st.locks)
    assert kinds == [("curtain", 12), ("ice", 12), ("ice", 15), ("ice", 17), ("ice", 20)]


def test_state_conveyor_ignores_uncertain_edge_slots(analyzer):
    st = analyzer.analyze(shot("033"), log=lambda s: None).state
    assert st.conveyor_k == 5
    assert all(b.fill is not None for b in st.conveyor)
    assert [b.color for b in st.conveyor] == ["orange", "cream", "darkred", "cream"]
