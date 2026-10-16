import json

import numpy as np
import pytest

pytest.importorskip("cv2")

from capture import Rect, ScreenGeometry
from config import ConfigError, from_dict
from safety import Hotkeys
from tools.collect import LABEL_RE, AutoTrigger, ShotWriter, shot_name

GEO = ScreenGeometry(Rect(273, 246, 317, 697), Rect(281, 280, 301, 651), 2.0)


def solid(v: int, h: int = 64, w: int = 32) -> np.ndarray:
    return np.full((h, w, 3), v, dtype=np.uint8)


def trigger(**kw) -> AutoTrigger:
    args = dict(change_threshold=0.03, stable_threshold=0.01, min_interval_s=1.0)
    args.update(kw)
    return AutoTrigger(**args)


def test_first_stable_frame_is_saved():
    t = trigger()
    assert t.feed(solid(100), 0.0) is None  # no previous frame yet
    assert t.feed(solid(100), 0.2) == 1.0   # stable, nothing saved before


def test_unstable_frames_are_skipped():
    t = trigger()
    t.feed(solid(0), 0.0)
    assert t.feed(solid(200), 0.2) is None  # big jump between consecutive frames = transition


def test_near_duplicate_of_saved_shot_is_skipped():
    t = trigger()
    t.feed(solid(100), 0.0)
    t.mark_saved(solid(100), 0.0)
    t.feed(solid(101), 5.0)
    assert t.feed(solid(101), 5.2) is None  # 1/255 < change_threshold


def test_new_screen_is_saved_after_it_settles_and_interval_passes():
    t = trigger()
    t.feed(solid(100), 0.0)
    t.mark_saved(solid(100), 0.0)
    assert t.feed(solid(200), 0.5) is None   # transition frame
    assert t.feed(solid(200), 0.7) is None   # settled, but within min_interval_s
    change = t.feed(solid(200), 1.2)
    assert change == pytest.approx(100 / 255, abs=1e-3)


def test_shape_change_resets_comparison():
    t = trigger()
    t.feed(solid(100), 0.0)
    t.mark_saved(solid(100), 0.0)
    assert t.feed(solid(100, h=80), 2.0) is None  # different size than previous frame
    assert t.feed(solid(100, h=80), 2.2) == 1.0


def test_shot_name_is_sortable_and_tagged():
    a = shot_name(1_790_000_000.25, "auto", None, 1)
    b = shot_name(1_790_000_001.0, "snap", "win", 2)
    assert a.endswith("-250-001-auto.png")
    assert b.endswith("-002-win-snap.png")
    assert a < b


def test_writer_saves_png_and_index(tmp_path):
    import cv2

    w = ShotWriter(tmp_path / "screens")
    frame = solid(123)
    shot = w.save(frame, GEO, "auto", "start", 0.5, stamp=1_790_000_000.0)
    assert shot.path.exists() and w.count == 1
    assert (cv2.imread(str(shot.path)) == frame).all()
    rec = json.loads((tmp_path / "screens" / "index.jsonl").read_text().strip())
    assert rec["file"] == shot.path.name and rec["label"] == "start" and rec["source"] == "auto"
    assert rec["frame_px"] == [32, 64] and rec["scale"] == 2.0
    assert rec["screen_pt"] == [281, 280, 301, 651]


@pytest.mark.parametrize("label,ok", [("win", True), ("mid_level-2", True), ("Win", False),
                                      ("../x", False), ("", False)])
def test_label_validation(label, ok):
    assert bool(LABEL_RE.match(label)) is ok


def test_collect_config_defaults_and_validation():
    assert from_dict({}).collect.hotkey_snap == "ctrl+alt+s"
    with pytest.raises(ConfigError):
        from_dict({"collect": {"change_threshold": 0}})
    with pytest.raises(ConfigError):
        from_dict({"collect": {"poll_s": 0}})


def test_hotkeys_extra_specs_are_parsed():
    Hotkeys("ctrl+alt+p", "ctrl+alt+q", extra={"ctrl+alt+s": lambda: None})
    with pytest.raises(ValueError):
        Hotkeys("ctrl+alt+p", "ctrl+alt+q", extra={"ctrl+alt+nope": lambda: None})
