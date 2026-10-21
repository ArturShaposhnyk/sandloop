"""Simulator speed-up (redesign phase B): LevelState.fork instead of deepcopy, cached contact
counts / colour totals in sim.Picture - same results as before, and clones never share state
they change."""
import copy
import random

from sim import Picture, Sim, SimConfig, grid_signature
from state import ConveyorBucket
from tests.test_state import make_state


def _walk(st, rnd, n=8):
    """Random tap sequences; the boards / locks / belt after every tap."""
    out = []
    for _ in range(n):
        if not st.tappable:
            break
        st = st.after_tap(rnd.choice(sorted(st.tappable)))
        out.append((grid_signature(st.grid), [(l.kind, l.counter, l.cells) for l in st.locks],
                    [(b.color, b.fill) for b in st.conveyor], sorted(st.tappable),
                    st.unlocked, st.chests_opened, st.revealed))
    return out


BOARDS = [
    ["red blue key:yellow cyan", "ice3 pink_ lock:yellow green_", "orange_ chest2 red_ mystery"],
    ["x red . tunnel3:left", "blue_ cyan_ red_ pink_", "chain:magenta green_ key:magenta x"],
]


def test_fork_gives_the_same_states_as_deepcopy(monkeypatch):
    from state import LevelState

    for rows in BOARDS:
        for seed in range(30):
            mk = lambda: make_state(rows, k=1, conveyor=[ConveyorBucket("cyan", 40)])  # noqa: E731
            new = _walk(mk(), random.Random(seed))
            with monkeypatch.context() as m:              # the old behaviour: deepcopy
                m.setattr(LevelState, "fork", lambda self: copy.deepcopy(self))
                old = _walk(mk(), random.Random(seed))
            assert new == old and new


def test_after_tap_never_changes_the_state_it_came_from():
    st = make_state(BOARDS[0], k=0)
    before = (grid_signature(st.grid), [(l.kind, l.counter) for l in st.locks])
    for p in list(st.tappable):
        st.after_tap(p)
    assert (grid_signature(st.grid), [(l.kind, l.counter) for l in st.locks]) == before


def test_a_curtain_counting_down_in_one_branch_leaves_its_siblings():
    """Sim.clone shares the grid; a fill that counts a curtain down must not change it for the
    parent or the other branches (it did: _fill_curtains edited the shared locks)."""
    from state import Lock

    st = make_state(["red red", "cur2 cur2"], k=0)   # curtain cells: no lock of their own
    st.locks.append(Lock("curtain", "fills", 2, ((1, 0), (1, 1))))
    pic = Picture(["red"], [[0] * 40 for _ in range(10)], contact=4)
    root = Sim.start(st, pic, SimConfig(fill_rate=500, fill_max=10, fill_min_exposed=0.0))
    a = root.tap((0, 0))                             # 10 %/s: not full during the tap
    b = a.clone()
    w = a.wait()                                     # the red bucket fills: curtain 2 -> 1
    assert [l.counter for l in w.grid.locks if l.kind == "curtain"] == [1]
    assert [l.counter for l in a.grid.locks if l.kind == "curtain"] == [2]
    assert [l.counter for l in b.grid.locks if l.kind == "curtain"] == [2]
    assert [l.counter for l in root.grid.locks if l.kind == "curtain"] == [2]


def test_picture_contact_counts_follow_every_take():
    rnd = random.Random(1)
    for _ in range(200):
        cols = [[rnd.randrange(4) for _ in range(rnd.randrange(0, 12))]
                for _ in range(rnd.randrange(1, 10))]
        p = Picture(["a", "b", "c", "d"], cols, rnd.randrange(1, 5))
        totals = {c: p.count(c) for c in "abcd"}
        for _ in range(20):
            c = rnd.choice("abcd")
            totals[c] -= p.take(c, rnd.randrange(0, 6))
            fresh = Picture(p.names, [list(x) for x in p.cols], p.contact)
            assert [p.share(x) for x in "abcd"] == [fresh.share(x) for x in "abcd"]
            assert {x: p.count(x) for x in "abcd"} == totals == {x: fresh.count(x) for x in "abcd"}
