import numpy as np
import pytest

pytest.importorskip("cv2")

from tools.overlay import BAR_H, clear_bar, waiting_frame


def test_clear_bar_blanks_only_the_bottom_strip():
    img = np.full((100, 50, 3), 200, dtype=np.uint8)
    clear_bar(img)
    assert (img[-BAR_H:] == 0).all()
    assert (img[: 100 - BAR_H] == 200).all()


def test_clear_bar_handles_images_shorter_than_the_bar():
    img = np.full((10, 5, 3), 200, dtype=np.uint8)
    clear_bar(img, height=30)
    assert (img == 0).all()


def test_waiting_frame_shape_and_text():
    img = waiting_frame("no on-screen window owned by 'iPhone Mirroring'. " * 5, size=(480, 220))
    assert img.shape == (220, 480, 3) and img.dtype == np.uint8
    assert (img > 128).any()  # some white text was drawn


def test_save_report_raw_only_without_dry_run(tmp_path):
    from tools.overlay import save_report

    img = np.full((40, 20, 3), 90, dtype=np.uint8)
    paths = save_report(tmp_path, img, None, stamp="t")
    assert [p.name for p in paths] == ["report-t.png"] and paths[0].exists()


def test_save_report_with_dry_run_decision(tmp_path):
    """The report must carry what is needed to explain a bad move: raw + annotated frame and
    the decision with every candidate."""
    from tests.test_state import shot
    from analyzer import Analyzer
    from tools.overlay import DryRun, save_report
    from config import load_config
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config.yaml")
    cfg.paths.palette = str(root / cfg.paths.palette)
    cfg.paths.dataset = str(root / cfg.paths.dataset)
    img = shot("052")
    dry = DryRun(Analyzer(cfg).analyze, interval_s=0, stable_below=1.0)
    dry.feed(img, diff=0.0)
    dry._future.result()
    dry.feed(img, diff=1.0)          # collect the result, no new analysis
    paths = save_report(tmp_path, img, dry, stamp="t")
    assert sorted(p.name for p in paths) == ["report-t-analysed.png", "report-t-annotated.png",
                                             "report-t.png", "report-t.txt"]
    text = (tmp_path / "report-t.txt").read_text()
    assert "decision: tap r2c3 darkred" in text and "r3c2 red: no" in text and "grid:" in text
    assert "avail" in text
    dry.close()


def test_save_report_fails_loudly(tmp_path):
    from tools.overlay import save_report

    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(OSError):
        save_report(blocker / "sub", np.zeros((4, 4, 3), np.uint8), None, stamp="t")
