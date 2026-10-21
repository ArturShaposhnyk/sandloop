"""Run debug/runs/20261021-143729 (L190, after the stall-clock hotfix): stopped by "2 taps in a
row had no effect" - r6c2 (cyan) was read "cyan/yellow" and tapped twice at 390 / 396 s.
The yellow hint arrows over the bottom row bob down to 0.887 of the height, below the
search zone (grid_zone ends at 0.88): not found, not masked, the arrow read as a lower tower
stage, and the tap point sat 7 px above the arrows' box (they bob ~20 px). Also frequent
"two objects in one cell" (a bucket flying up its lane, a tunnel's drop): saved as unknown.
"""
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

from runlog import load_events  # noqa: E402
from tests.test_bot import make_bot, run  # noqa: E402
from tests.test_bot import analyzer, cfg  # noqa: E402,F401  (fixtures)

DATA = Path(__file__).resolve().parent / "data" / "run-20261021-143729"


def frame(name):
    img = cv2.imread(str(DATA / f"{name}.png"))
    assert img is not None, name
    return img


def test_arrows_below_the_grid_are_masked(analyzer):  # noqa: F811
    img = frame("00475-r6c2-under-arrows")
    res = analyzer.analyze(img, log=lambda s: None, now=0.0)
    assert len(res.view.hints) == 4
    cell = res.state.cell((6, 2))
    assert (cell.color, cell.body, cell.face_up) == ("cyan", None, True)
    # the tap point is above the arrows' highest position (box top - 20 px of bobbing)
    _, y = res.view.cell_center_rel(6, 2)
    top = min(b[1] for b in res.view.hints)
    assert y * img.shape[0] < top - 20


def test_a_tap_without_effect_is_not_repeated_on_the_same_board(cfg, analyzer, tmp_path):  # noqa: F811
    img = frame("00475-r6c2-under-arrows")
    bot, log = make_bot(cfg, analyzer, tmp_path, [img] * 30)
    try:
        run(bot, log)
    except Exception:
        pass
    taps = [tuple(e["pos"]) for e in load_events(log.dir) if e["type"] == "tap"]
    assert taps and len(taps) == len(set(taps))


def test_flying_bucket_is_no_new_mechanic(analyzer, tmp_path):  # noqa: F811
    img = frame("00030-r1c0-bucket-flying")
    analyzer.unknown_dir = tmp_path
    analyzer.cells_seen = set()
    analyzer._two_since = {}
    res = analyzer.analyze(img, log=lambda s: None, now=100.0)
    assert any(c.reason.startswith("two objects") for c in res.view.unknown_cells())
    analyzer.analyze(img, log=lambda s: None, now=101.0)
    assert not list(tmp_path.glob("*.png"))          # one moment: not saved
    analyzer.analyze(img, log=lambda s: None, now=102.5)
    assert list(tmp_path.glob("*.png"))              # still there after 2 s: saved
