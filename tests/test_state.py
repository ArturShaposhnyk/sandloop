"""state.py on synthetic grids (pure logic) and on saved screenshots."""
from typing import Dict, List, Optional

import pytest

from cells import Cell, Kind
from state import ConveyorBucket, LevelState, Lock


def make_state(rows: List[str], k: Optional[int] = 0, conveyor=(), sand: Optional[Dict] = None,
               exposed: Optional[Dict] = None, locks=()) -> LevelState:
    """Rows of space-separated cells: '.' empty, 'x' no cell, 'ice3' lock, 'cur2' curtain cell,
    'red' face-up bucket, 'red_' face-down bucket, 'red/blue' tower (top/body),
    'lock:red' / 'key:red' lock icon / key."""
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
            elif t.startswith(("lock:", "key:")):
                kind, _, colour = t.partition(":")
                out.append(Cell(r, c, Kind.LOCK, lock=kind,
                                unlock="key" if kind == "lock" else "path", color=colour))
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
                locks.append(Lock(cell.lock, cell.unlock, cell.counter, ((cell.row, cell.col),),
                                  cell.color))
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


def test_tower_tap_sends_the_whole_tower_to_the_belt():
    """GAME_RULES: the whole tower leaves the grid (cell empty at once) and takes ONE belt slot;
    its stages collect one after another on the belt."""
    st = make_state(["red/blue",
                     "pink_"], k=1, conveyor=[ConveyorBucket("cyan", 40)])
    nxt = st.after_tap((0, 0))
    assert nxt.cell((0, 0)).kind is Kind.EMPTY
    assert nxt.conveyor_k == 2                                   # one slot for both stages
    assert nxt.conveyor[-1] == ConveyorBucket("red", 0, stages=("blue",))
    assert nxt.tappable == [(1, 0)]                              # pink is connected now
    assert st.opened_by((0, 0)) == ([(1, 0)], [])
    assert nxt.conveyor_need("red") == 1.0 and nxt.conveyor_need("blue") == 1.0


def test_belt_tower_stages_count_as_demand_for_their_colour():
    b = ConveyorBucket("cream", 34, stages=("red",))
    assert b.need_for("cream", 0.5) == pytest.approx(0.66)
    assert b.need_for("red", 0.5) == 1.0 and b.need_for("blue", 0.5) == 0.0
    st = make_state(["red"], k=2, conveyor=[b, ConveyorBucket("red", 50)])
    assert st.conveyor_need("red") == pytest.approx(1.5)
    assert st.pending_buckets() == pytest.approx(1 + 0.66 + 1 + 0.5)


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


def test_belt_tracker_stall_clock():
    from state import BeltTracker

    t = BeltTracker()
    red = lambda f: ConveyorBucket("red", f)  # noqa: E731
    assert t.update([red(40)], 0.0) == {"red": 0.0}
    assert t.update([red(40)], 3.0) == {"red": 3.0}          # no growth
    assert t.update([red(45)], 4.0) == {"red": 0.0}          # grew: clock restarts
    assert t.update([], 6.0) == {}                           # out of sight: clock kept
    assert t.update([red(45)], 9.0) == {"red": 5.0}
    assert t.update([red(45), red(0)], 10.0) == {"red": 0.0}  # a new red bucket arrived
    assert t.update([red(100), ConveyorBucket("blue", None)], 11.0) == {}  # done / unread


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
    path, = [p for p in root.glob("*.png") if p.stem.split("-")[3] == seq]  # exactly one
    return _cv2.imread(str(path))


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


# -- locks and keys (L183, docs/GAME_RULES.md) ----------------------------------------------------

def test_key_connected_by_a_tap_flies_and_opens_its_lock():
    st = make_state(["red blue x",
                     "key:yellow pink_ lock:yellow",
                     "lock:red cyan_ cyan_"])
    assert st.hidden_buckets == 2                 # one under every lock cell
    nxt = st.after_tap((0, 0))                    # connects the yellow key to the exit
    assert nxt.unlocked == [("yellow", ((1, 2),))]
    assert nxt.cell((1, 0)).kind is Kind.EMPTY    # the key flew away (assumption)
    assert nxt.cell((1, 2)).kind is Kind.UNKNOWN  # lock open, bucket colour not seen yet
    assert nxt.cell((2, 0)).kind is Kind.LOCK     # the red lock stays: its key is not here
    assert [l.kind for l in nxt.locks] == ["lock"]
    assert st.cell((1, 0)).kind is Kind.LOCK      # original state untouched


def test_key_release_chains_and_unconnected_key_stays():
    st = make_state(["red pink_ x",
                     "key:blue lock:red green_",
                     "key:red lock:blue cyan_"])
    nxt = st.after_tap((0, 0))                    # blue key flies, which connects the red key
    assert nxt.unlocked == [("blue", ((2, 1),)), ("red", ((1, 1),))]
    st2 = make_state(["red blue",
                      "pink_ key:red",
                      "lock:red cyan_"])
    assert st2.after_tap((0, 0)).unlocked == []   # the key is still boxed in
    assert st2.after_tap((0, 1)).unlocked == [("red", ((2, 0),))]


def test_key_already_connected_is_no_credit_for_a_tap():
    st = make_state(["key:red blue",
                     "lock:red pink_"])
    nxt = st.after_tap((0, 1))
    assert nxt.unlocked == [] and nxt.cell((0, 0)).kind is Kind.EMPTY


def test_adjacent_lock_cells_of_one_colour_form_one_lock():
    from state import _merge_lock_icons

    locks = [Lock("lock", "key", None, ((1, 1),), "red"), Lock("lock", "key", None, ((1, 2),), "red"),
             Lock("lock", "key", None, ((3, 3),), "blue"), Lock("ice", "taps", 2, ((0, 0),))]
    merged = _merge_lock_icons(locks)
    assert sorted((l.kind, l.color, l.cells) for l in merged if l.kind == "lock") == [
        ("lock", "blue", ((3, 3),)), ("lock", "red", ((1, 1), (1, 2)))]


def test_strategy_credits_the_lock_a_tap_opens():
    from config import StrategyConfig
    from strategy import choose

    st = make_state(["red blue",
                     "key:yellow pink_",
                     "lock:yellow cyan_"], sand={"red": 0.3, "blue": 0.3},
                    exposed={"red": 0.2, "blue": 0.2})
    d = choose(st, StrategyConfig())
    red = next(c for c in d.candidates if c.color == "red")
    assert red.unlocks == [("yellow", ((2, 0),))] and "opens lock yellow with 1 bucket" in red.describe()
    assert d.move.color == "red"                  # same exposure, but it releases the key
