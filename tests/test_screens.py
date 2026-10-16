"""Screen classification on every labeled screenshot (dataset/screens/index.jsonl labels)."""
import json
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

from config import load_config
from screens import RULES, SCREEN_TYPES, ScreenClassifier, load_templates

ROOT = Path(__file__).resolve().parents[1]
SCREENS = ROOT / "dataset" / "screens"
RECORDS = [json.loads(l) for l in (SCREENS / "index.jsonl").read_text().splitlines() if l.strip()]

KNOWN_LIMITATIONS = {
    "020": "ad end card without Get App / mute button: no anchor, classified 'other' "
           "(the bot waits there either way)",
}


@pytest.fixture(scope="module")
def clf():
    cfg = load_config(ROOT / "config.yaml")
    return ScreenClassifier(load_templates(ROOT / "dataset" / "templates"), cfg.vision)


@pytest.mark.parametrize("rec", RECORDS, ids=[r["file"].split("-")[3] for r in RECORDS])
def test_screen_type(rec, clf):
    seq = rec["file"].split("-")[3]
    if seq in KNOWN_LIMITATIONS:
        pytest.xfail(KNOWN_LIMITATIONS[seq])
    assert clf.classify(cv2.imread(str(SCREENS / rec["file"]))).screen == rec["label"]


def test_rules_only_use_known_types_and_templates(clf):
    for screen, anchors in RULES:
        assert screen in SCREEN_TYPES
        assert all(a in clf.templates for a in anchors), anchors


def test_close_button_location_on_popup(clf):
    rec = next(r for r in RECORDS if r["file"].split("-")[3] == "024")
    res = clf.classify(cv2.imread(str(SCREENS / rec["file"])))
    x, y = res.find("close_red").center_rel
    assert 0.85 < x < 0.95 and 0.02 < y < 0.07


def test_black_frame_is_other(clf):
    import numpy as np

    assert clf.classify(np.zeros((1302, 602, 3), np.uint8)).screen == "other"
