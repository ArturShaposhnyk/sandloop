"""L188 shells / picture locks and L189 chain locks (user, 2026-10-21; docs/GAME_RULES.md).

Frames: live run debug/runs/20261021-001541 (L188 start, the grid failed "walls off its cell
boundaries (fit 0.77)"; tests/data/run-20261021-001541/), the user's L188 game recorded as
dataset shots 065-071 and the L189 start screenshot (shot 072, a Mac capture).
"""
import glob
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

from cells import Kind  # noqa: E402
from picture import PicLock, PicShell  # noqa: E402
from sim import Picture, Shell, Sim, SimConfig  # noqa: E402
from state import ConveyorBucket  # noqa: E402
from strategy import choose, evaluate  # noqa: E402
from tests.test_state import analyzer, make_state  # noqa: E402,F401  (fixture)
from vision import find_lattice, read_gameplay  # noqa: E402

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "data" / "run-20261021-001541"
SHOTS = ROOT.parent / "dataset" / "screens"


def shot(seq):
    (path,) = glob.glob(str(SHOTS / f"*-{seq}-*.png"))
    return cv2.imread(path)


def view(an, img):
    return read_gameplay(img, an.palette, an.cfg.vision, an.reader, log=lambda s: None,
                         icons=an.icons, hints=an.hints)


def state_of(an, img):
    from state import LevelState

    return LevelState.from_view(view(an, img), 5, an.cfg.strategy.unknown_fill_need)


# --- L188 grid / shell buckets -----------------------------------------------------------------

def test_l188_lattice_fits(analyzer):  # noqa: F811
    """Bucket pixels near the panel colour scattered short edges inside the columns (0.77)."""
    lat = find_lattice(cv2.imread(str(RUN / "00001-wait.png")), analyzer.palette,
                       analyzer.cfg.vision)
    assert (lat.rows, lat.cols) == (6, 4) and lat.fit >= analyzer.cfg.vision.lattice_min_fit


@pytest.mark.parametrize("seq, cells", [
    ("068", {(2, 0): "pink-up+shell", (2, 3): "pink-up+shell", (3, 0): "blue-up+shell",
             (5, 3): "darkred-up+shell", (1, 2): "darkgreen-up"}),
    ("069", {(2, 0): "pink-up+shell", (2, 3): "pink-up+shell", (5, 3): "darkred-up+shell"}),
])
def test_open_shell_buckets(analyzer, seq, cells):  # noqa: F811
    """Open shell buckets show a GREY shell (read mystery / cream stage before)."""
    gv = view(analyzer, shot(seq))
    for (r, c), label in cells.items():
        assert gv.grid[r][c].label() == label, (r, c)


def test_l189_plain_dome_is_no_shell(analyzer):  # noqa: F811
    gv = view(analyzer, shot("072"))
    assert not any(c is not None and c.shell for row in gv.grid for c in row)


# --- L188 picture -----------------------------------------------------------------------------

def test_shells_and_locks_at_the_start(analyzer):  # noqa: F811
    gv = view(analyzer, shot("065"))
    colours = sorted(s.colour for s in gv.shells)
    assert colours == ["blue", "blue", "darkred", "darkred", "pink", "pink"]
    assert all(s.depth > analyzer.cfg.strategy.shell_near_bottom for s in gv.shells)
    locks = sorted((l.colour, l.rect) for l in gv.pic_locks)
    assert [c for c, _ in locks] == ["red", "yellow"]
    (_, red), (_, yellow) = locks
    assert yellow == pytest.approx((0.171, 0.334, 0.383, 0.432), abs=0.01)
    assert red == pytest.approx((0.617, 0.334, 0.829, 0.432), abs=0.01)
    assert 0.1 < gv.locked < 0.3
    # the boxes' overlay is out of the sand reading: no unknown colour, no "cream 64 %"
    assert not gv.unknown_colours and gv.exposed.get("cream", 0) < 0.3


def test_locks_gone_and_shells_down_later(analyzer):  # noqa: F811
    near = analyzer.cfg.strategy.shell_near_bottom
    gv = view(analyzer, shot("069"))
    assert not gv.pic_locks and gv.locked == 0
    down = {s.colour for s in gv.shells if s.depth <= near}
    assert down == {"blue", "darkred"}
    assert all(s.depth > near for s in gv.shells if s.colour == "pink")


def test_shell_bucket_waits_for_its_shell(analyzer):  # noqa: F811
    """069: the darkred shell is at the bottom (r5c3 may go), the pink ones 6 % up (wait)."""
    st = state_of(analyzer, shot("069"))
    sc = analyzer.cfg.strategy
    pink = evaluate(st, (2, 0), sc)
    assert pink.shell_wait and pink.rejected.startswith("shell bucket: no pink shell")
    assert not evaluate(st, (5, 3), sc).shell_wait


def test_unknown_mechanic_notes(analyzer):  # noqa: F811
    logs = []
    analyzer.reset_level()
    analyzer.analyze(shot("065"), log=logs.append, now=0.0)
    assert any("picture lock yellow: its key is not on the board yet (under a curtain" in s
               for s in logs)
    analyzer.reset_level(track=False)


def test_key_opens_its_picture_lock():
    st = make_state(["red .", "key:yellow blue"])
    st.pic_locks = [PicLock("yellow", (0.2, 0.3, 0.4, 0.4))]
    after = st.after_tap((0, 0))     # the key's cell connects -> it flies to the picture lock
    assert after.pics_opened == ["yellow"] and not after.pic_locks


# --- simulator --------------------------------------------------------------------------------

def test_sim_shell_falls_and_drops_into_its_bucket():
    pic = Picture(["red", "pink"], [[0, 0, 0, 0, 1]], contact=2, shells=[Shell("pink", 0, 4)])
    assert pic.shell_down("pink") is None
    assert pic.take("red", 2) == 2 and pic.shells[0].below == 2
    assert pic.shell_down("pink") is not None
    st = make_state(["pink+shell"])
    st.shells = [PicShell("pink", 0.5, 0.4, 0.0)]
    sim = Sim.start(st, pic, SimConfig(tap_s=0.0), {"pink": 1, "red": 4})
    assert sim.moves() == [(0, 0)]
    nxt = sim.tap((0, 0))
    assert nxt.belt[0].shell                      # waiting for its shell
    nxt.lap()
    assert not nxt.belt or not nxt.belt[0].shell  # the shell dropped in
    assert not nxt.pic.shells


def test_sim_shell_bucket_not_tapped_while_its_shell_is_high():
    pic = Picture(["red", "pink"], [[0, 0, 0, 0, 1]], contact=2, shells=[Shell("pink", 0, 4)])
    st = make_state(["pink+shell red"])
    sim = Sim.start(st, pic, SimConfig(tap_s=0.0), {"pink": 1, "red": 2})
    assert sim.moves() == [(0, 1)]


# --- L189 chain lock --------------------------------------------------------------------------

def test_l189_reads_the_chain_lock(analyzer):  # noqa: F811
    st = state_of(analyzer, shot("072"))
    (gate,) = [l for l in st.locks if l.kind == "chain"]
    assert gate.color == "yellow" and gate.cells == ((4, 2), (5, 2))
    keys = sorted((l.color, l.cells) for l in st.locks if l.kind == "key")
    assert keys == [("magenta", ((5, 0),)), ("yellow", ((5, 6),))]


def test_chain_gate_blocks_until_its_key_flies():
    st = make_state([". . .",
                     "x chain:yellow x",
                     "x red_ x"])
    assert (2, 1) not in st.connected()           # a gate: impassable like a wall
    st = make_state(["red .",
                     "key:yellow chain:yellow",
                     "x chain:?",
                     "x blue_"])
    (gate,) = [l for l in st.locks if l.kind == "chain"]
    assert gate.color == "yellow" and gate.cells == ((1, 1), (2, 1))
    assert st.behind_gate(gate) == [(3, 1)]
    after = st.after_tap((0, 0))                  # connects the key: it flies to the gate
    assert after.gates_opened == [("yellow", ((1, 1), (2, 1)))]
    assert after.cell((1, 1)).kind is Kind.EMPTY and (3, 1) in after.playable()


def test_planner_values_the_chain_key_by_what_is_behind(analyzer):  # noqa: F811
    from planner import targets

    st = make_state(["red red .",
                     "blue key:yellow chain:yellow",
                     "x x blue_"],
                    sand={"red": .2, "blue": .2}, exposed={"red": .2, "blue": .2})
    labels = {label: value for _, label, value in targets(st, analyzer.cfg.strategy)}
    assert "key yellow (chain: 1 bucket(s) behind)" in labels
