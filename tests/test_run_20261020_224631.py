"""Regressions from the live run debug/runs/20261020-224631 (L187 --levels 3: lost, then an ad
end card held the session until the quit hotkey).

* After Try Again: Kingshot ad, ▶| skip (tapped), App Store sheet ✕ (tapped), then an end card
  with a small white ✕ on a yellow disc top right. The ✕ is 0.017 x the screen width, below
  ``flow.close_min_size`` (kept there for the mute icon's x), and the disc finder only knew ▶|:
  100 s "unknown screen" until the user quit.

* Endgame: #24 the level plan tapped the orange/darkgreen tower r3c4 that the rules rejected
  (avail -5 %); both orange buckets stalled. #27 the last-slot gamble took purple r5c6 (3.2 %
  sand left, a bucket 3.67 %) because it uncovered the most for the belt. User: never a bucket
  that cannot fill on the last slot (unless it opens one that can), fillers first; a plan
  step the sand rules reject is skipped and re-planned; a tower only when its top can fill.

tests/data/run-20261020-224631/: 00427 (the end card), 00313 (the App Store sheet), 00228 (tap
#24), 00236 / 00243 (taps #25 / #26), 00262 (tap #27, k 4).
"""
import dataclasses
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

from adclose import find_close, find_skip  # noqa: E402
from flow import AD_SKIP, ScreenFlow  # noqa: E402
from endgame import key_reserve  # noqa: E402
from levelplan import Plan, PlanRunner, Step  # noqa: E402
from screens import ScreenResult  # noqa: E402
from sim import grid_signature  # noqa: E402
from state import LevelState  # noqa: E402
from strategy import choose, evaluate  # noqa: E402
from vision import read_gameplay  # noqa: E402
from tests.test_state import analyzer  # noqa: E402,F401  (fixture)

DATA = Path(__file__).resolve().parent / "data" / "run-20261020-224631"
# bot.log: "capacity per colour (share of the picture per bucket)" at level start
CAP = {"blue": .0301, "cream": .0089, "darkgreen": .0471, "darkred": .0119, "orange": .0402,
       "purple": .0367, "yellow": .0407}
END_STALLED = {"blue": 28.0, "orange": 10.0, "cream": 8.0}   # belt stuck before #27
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


# --- endgame ----------------------------------------------------------------------------------

def run_state(analyzer, name, stalled):  # noqa: F811
    gv = read_gameplay(frame(name), analyzer.palette, analyzer.cfg.vision, analyzer.reader,
                       log=lambda s: None, icons=analyzer.icons, hints=analyzer.hints)
    st = LevelState.from_view(gv, 5, analyzer.cfg.strategy.unknown_fill_need, stalled)
    st.colour_capacity = dict(CAP)
    return st


def test_tap_27_never_gambles_a_bucket_that_cannot_fill(analyzer):  # noqa: F811
    """Tappable: darkred r0c4 (1.1 % left), yellow r2c2 (3.8 %), purple r5c6 (3.2 %): none
    has a whole bucket of sand left (yellow 4.07 %), none opens a bucket -> no gamble (the run
    took purple). With ``gamble_fill_margin`` 0.9 yellow, which fills itself, would go."""
    sc = analyzer.cfg.strategy
    st = run_state(analyzer, "00262-tap", END_STALLED)
    d = choose(st, sc)
    assert d.move is None, d.reason
    d = choose(st, dataclasses.replace(sc, gamble_fill_margin=0.9))
    assert d.move.gamble and (d.move.pos, d.move.color) == ((2, 2), "yellow")
    assert d.move.fills == "fills itself"


def test_tap_24_tower_step_is_skipped_and_replanned(analyzer):  # noqa: F811
    st = run_state(analyzer, "00228-tap", {})
    sc = analyzer.cfg.strategy
    cand = evaluate(st, (3, 4), sc, opener=True)
    assert cand.stages == ["darkgreen"] and cand.low_exposure
    runner = PlanRunner(analyzer.cfg)
    board = grid_signature(st.grid)
    runner.signature, runner.last_plan_t, runner.log = "test", 0.0, [{"t": 0.0}]
    runner.plan = Plan([Step((3, 4), "orange", 2, board)], "open", 0, 0, 0)
    logs = []
    fallback = choose(st, sc)
    d = runner.decide(frame("00228-tap"), None, st, fallback, 1.0, logs.append)
    assert any("skipped: not enough exposed sand" in s for s in logs), logs
    assert runner.forbid[board] == {(3, 4)}
    assert all(s.pos != (3, 4) for s in runner.plan.steps if s.board == board)
    assert d.move is None or d.move.pos != (3, 4)


def test_tower_goes_only_when_its_top_can_fill(analyzer):  # noqa: F811
    """#24 even with orange exposed: 6.6 % orange left, the belt orange (23 %) still needs
    3.1 % and the tower's top a whole bucket (4.0 %) -> its darkgreen would never go."""
    st = run_state(analyzer, "00228-tap", {})
    sc = analyzer.cfg.strategy
    st.exposed["orange"] = 0.5                               # exposure no longer the issue
    cand = evaluate(st, (3, 4), sc)
    assert cand.sand_short and cand.rejected.startswith("tower: orange cannot fill")
    st.sand["orange"] = 0.08
    assert evaluate(st, (3, 4), sc).rejected is None


def test_tap_27_key_colour_takes_the_reserved_last_slot(analyzer):  # noqa: F811
    """The key reserve kept the last slot for yellow (22.9 % of the contact layer, covers blue /
    cream on the belt), then the last-slot minimum (25 %) rejected yellow itself. A key colour's
    own bucket now needs only the normal minimum there. The lookahead still calls it dead at
    look_sand_margin 1.0 (3.8 % yellow left, a bucket 4.07 %); at 0.9 it is tapped."""
    sc = analyzer.cfg.strategy
    st = run_state(analyzer, "00262-tap", END_STALLED)
    assert "keep 1 slot(s) to reach yellow r2c2" in key_reserve(st, (0, 4), sc)
    assert evaluate(st, (2, 2), sc).rejected is None
    # purple (21.7 %) is a key colour too; the endgame reserve keeps its slot for yellow
    assert evaluate(st, (5, 6), sc).rejected.startswith("endgame: keep a slot for yellow")
    yellow = next(c for c in choose(st, sc).candidates if c.pos == (2, 2))
    assert yellow.rejected.startswith("lookahead: the belt jams")
    d = choose(st, dataclasses.replace(sc, look_sand_margin=0.9))
    assert d.move is not None and (d.move.pos, d.move.color) == ((2, 2), "yellow")


def test_tap_26_yellow_was_not_a_key_yet(analyzer):  # noqa: F811
    """Before #26 (2 free): yellow 18.5 % < key_min_exposed 20 % and it covers only 0.2 % of
    the buried blue - no key; as a key it needs 1 slot, which cream would have left anyway."""
    from endgame import key_colours
    st = run_state(analyzer, "00243-tap", {"blue": 25.0})
    assert "yellow" not in key_colours(st, analyzer.cfg.strategy)


# --- ad chains --------------------------------------------------------------------------------

def _chain_flow(analyzer):  # noqa: F811
    fc = dataclasses.replace(analyzer.cfg.flow, confirm_frames=1, tap_cooldown_s=0.0)
    fl = ScreenFlow(fc)
    fl.last_tap = -10.0                                      # Try Again
    return fc, fl


def test_ad_chain_taps_as_many_ads_as_come(analyzer):  # noqa: F811
    """User: keep tapping ▶| / ✕ every few seconds until a known screen is back - more than
    same_button_max / max_taps_between_levels times."""
    fc, fl = _chain_flow(analyzer)
    other = ScreenResult("other", {})
    skip, card = cv2.imread(str(TRADE_REPUBLIC)), frame("00427-screen-other-wait")
    t, taps = 1.0, 0
    for i in range(int(3 * fc.max_taps_between_levels * fc.ad_retap_s)):
        act = fl.feed(other, skip if i % 2 else card, 5, t)
        assert act.kind != "stop", act.reason
        if act.kind == "tap":
            assert act.button == AD_SKIP
            fl.tapped(act, t)
            taps += 1
        t += 1.0
    assert taps > fc.max_taps_between_levels
    assert fl.taps_between == 0 and not fl.same               # ad taps are not counted


def test_ad_chain_retaps_every_few_seconds(analyzer):  # noqa: F811
    fc, fl = _chain_flow(analyzer)
    other, card = ScreenResult("other", {}), frame("00427-screen-other-wait")
    act = fl.feed(other, card, 5, 1.0)
    fl.tapped(act, 1.0)
    assert fl.feed(other, card, 5, 1.0 + fc.ad_retap_s / 2).kind == "wait"
    assert fl.feed(other, card, 5, 1.0 + fc.ad_retap_s + 0.1).kind == "tap"


def test_ad_chain_ends_on_a_known_screen_and_has_a_cap(analyzer):  # noqa: F811
    fc, fl = _chain_flow(analyzer)
    other, card = ScreenResult("other", {}), frame("00427-screen-other-wait")
    fl.tapped(fl.feed(other, card, 5, 1.0), 1.0)
    assert fl.chain_start == 1.0
    fl.feed(ScreenResult("fail", {}), card, 5, 5.0)          # LEVEL FAILED is back
    assert fl.chain_start is None
    fl.tapped(fl.feed(other, card, 5, 10.0), 10.0)
    act = fl.feed(other, card, 5, 10.0 + fc.ad_chain_max_s + 1)
    assert act.kind == "stop" and "ad chain" in act.reason
