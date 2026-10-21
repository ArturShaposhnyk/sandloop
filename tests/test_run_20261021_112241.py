"""Run debug/runs/20261021-112241: L189 WON (attempt 2). Then a false "LEVEL 3: WIN (inferred:
sand left 0.0%, then screen other)": after the win, NEXT and the "take a break" ad, the finished
L189 board faded back in (frame 00470 here, from dataset/unknown/20261021-112948-707): no sand,
its empty cells dimmed (LAB 80/-1/-13 -> "two objects" / unknown colours). That frame started a
level and the next screen change counted it as won.

Also the L190 tunnel (dispenser) mechanic (user, 2026-10-21) on synthetic boards - no L190 frame
exists yet (the run stopped at the phantom win).
"""
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

from cells import CELL_LOCK_BY_NAME, CellFeatures, Kind, classify_cell  # noqa: E402
from runlog import load_events  # noqa: E402
from sim import Picture, Sim, SimConfig  # noqa: E402
from tests.test_bot import Tapper, make_bot, run  # noqa: E402
from tests.test_bot import analyzer, cfg  # noqa: E402,F401  (fixtures)
from tests.test_state import make_state  # noqa: E402

DATA = Path(__file__).resolve().parent / "data" / "run-20261021-112241"


def frame(name):
    img = cv2.imread(str(DATA / f"{name}.png"))
    assert img is not None, name
    return img


def test_faded_finished_board_starts_no_level(cfg, analyzer, tmp_path):  # noqa: F811
    faded, other = frame("00470-finished-board-faded"), frame("00475-screen-other")
    bot, log = make_bot(cfg, analyzer, tmp_path, [faded] * 3 + [other] * 4, tapper=Tapper())
    run(bot, log)
    events = load_events(log.dir)
    assert not [e for e in events if e["type"] in ("level_start", "level_end")]
    assert any(e["type"] == "finished_board" for e in events)


def test_inferred_win_needs_taps_and_sand_at_the_start():
    from types import SimpleNamespace

    from levels import LevelMixin

    ended = []
    log = SimpleNamespace(frame=lambda *a: "x.png", log=lambda *a: None,
                          event=lambda *a, **k: ended.append(k), t=lambda: 0.0)

    def bot(taps, start_sand):
        b = SimpleNamespace(last_sand=0.0, bc=SimpleNamespace(win_sand_max=0.01,
                                                              infer_win_start_sand=0.05),
                            log=log, level_no=1, frame_no=1, analyzer=None, taps=0,
                            level_taps=taps, level_first={"sand_total": start_sand},
                            left_unclear=False, in_level=True,
                            session=SimpleNamespace(record=lambda r: None))
        b._record_level = lambda result, how: None
        return b

    for b in (bot([], 0.0), bot([{"n": 1}], 0.0), bot([], 0.8)):   # the phantom level, ...
        LevelMixin._level_left(b, SimpleNamespace(screen="other"), None)
        assert b.in_level and not ended
    b = bot([{"n": 1}], 0.8)                                         # a real level emptied
    LevelMixin._level_left(b, SimpleNamespace(screen="other"), None)
    assert not b.in_level and ended[-1]["result"] == "win"


# --- L190 tunnel (dispenser) ---------------------------------------------------------------------

def test_tunnel_refills_the_cell_in_front_of_its_opening():
    st = make_state(["red blue", "tunnel2:up green"])
    tun = next(l for l in st.locks if l.kind == "tunnel")
    assert tun.front() == (0, 0) and st.hidden_buckets == 2
    after = st.after_tap((0, 0))                     # red taken: the next bucket drops in
    cell = after.cell((0, 0))
    assert cell.kind is Kind.UNKNOWN and cell.hidden == 1
    assert next(l for l in after.locks if l.kind == "tunnel").counter == 1
    assert after.cell((1, 0)).label() == "tunnel1:up"
    other = st.after_tap((0, 1))                     # not in front: nothing drops
    assert next(l for l in other.locks if l.kind == "tunnel").counter == 2


def test_empty_tunnel_drops_nothing_and_stays():
    st = make_state(["red", "tunnel0:up"])
    after = st.after_tap((0, 0))
    assert after.cell((0, 0)).kind is Kind.EMPTY and after.cell((1, 0)).kind is Kind.LOCK


def test_tunnel_is_a_wall_for_the_path_rule():
    st = make_state(["tunnel3:down", "red"])         # red's only neighbour toward the exit
    assert (1, 0) not in st.playable()


def test_planner_treats_the_drop_as_unknown():
    st = make_state(["red", "tunnel3:up"])
    pic = Picture(["red"], [[0, 0, 0]] * 4, contact=1)
    s = Sim.start(st, pic, SimConfig(slots=5)).tap((0, 0))
    assert s.grid.cell((0, 0)).kind is Kind.UNKNOWN and s.unknown_left()
    assert s.moves() == []                          # the dropped colour is not known


def test_tunnel_icon_classifies_with_its_opening(analyzer):  # noqa: F811
    v = analyzer.cfg.vision
    f = CellFeatures(empty_frac=0.0, white_frac=0.0, dark_frac=0.0, face_ratio=1.0, top_lab=None,
                     body_lab=None, hue_cyan_frac=0.0, icon="tunnel_left")
    cell = classify_cell(2, 3, f, analyzer.palette, v)
    assert (cell.kind, cell.lock, cell.opening) == (Kind.LOCK, "tunnel", "left")
    assert CELL_LOCK_BY_NAME["tunnel"].counter == "optional"


def test_annotation_label_parses():
    from tools.build_palette import parse_label

    assert parse_label("tunnel5:down") == ("tunnel", "down", None, None, 5)
