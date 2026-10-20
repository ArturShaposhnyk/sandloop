"""Regressions from the live run debug/runs/20261020-224631 (L187 --levels 3: lost, then an ad
end card held the session until the quit hotkey).

* After Try Again: Kingshot ad, ▶| skip (tapped), App Store sheet ✕ (tapped), then an end card
  with a small white ✕ on a yellow disc top right. The ✕ is 0.017 x the screen width, below
  ``flow.close_min_size`` (kept there for the mute icon's x), and the disc finder only knew ▶|:
  100 s "unknown screen" until the user quit.

tests/data/run-20261020-224631/: 00427 (the end card), 00313 (the App Store sheet).
"""
import dataclasses
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

from adclose import find_close, find_skip  # noqa: E402
from flow import AD_SKIP, ScreenFlow  # noqa: E402
from screens import ScreenResult  # noqa: E402
from tests.test_state import analyzer  # noqa: E402,F401  (fixture)

DATA = Path(__file__).resolve().parent / "data" / "run-20261020-224631"
TRADE_REPUBLIC = (Path(__file__).resolve().parent / "data" / "run-20261020-171133"
                  / "00301-before-window-lost.png")


def frame(name):
    img = cv2.imread(str(DATA / f"{name}.png"))
    assert img is not None, name
    return img


def test_small_cross_on_a_disc_closes_the_end_card(analyzer):  # noqa: F811
    img = frame("00427-screen-other-wait")
    assert find_close(img, analyzer.cfg.flow) == []          # too small for the generic ✕
    (mark,) = find_skip(img, analyzer.cfg.flow)
    assert mark.kind == "close" and mark.score >= analyzer.cfg.flow.close_min_iou
    assert mark.center_rel[0] > 0.93 and mark.center_rel[1] < 0.1


def test_skip_disc_still_reads_as_skip(analyzer):  # noqa: F811
    (mark,) = find_skip(cv2.imread(str(TRADE_REPUBLIC)), analyzer.cfg.flow)
    assert mark.kind == "skip"


def test_app_store_sheet_has_no_disc_button(analyzer):  # noqa: F811
    img = frame("00313-screen-tap-ad_close")
    assert find_skip(img, analyzer.cfg.flow) == []
    assert find_close(img, analyzer.cfg.flow)[0].center_rel[0] < 0.2   # its own ✕, top left


def test_end_card_after_an_ad_tap_is_closed(analyzer):  # noqa: F811
    fc = dataclasses.replace(analyzer.cfg.flow, confirm_frames=1, tap_cooldown_s=0.0)
    fl = ScreenFlow(fc)
    fl.last_tap = 326.8                                      # the App Store sheet's ✕
    act = fl.feed(ScreenResult("other", {}), frame("00427-screen-other-wait"), 5, 330.0)
    assert act.kind == "tap" and act.button == AD_SKIP and "✕ on a disc" in act.reason
