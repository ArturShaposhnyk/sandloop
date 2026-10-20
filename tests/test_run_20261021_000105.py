"""Regressions from the live run debug/runs/20261021-000105 - the first L187 (HARD) win (29 taps,
the level plan followed to the end, belt speed 2x switched on at the start).

Bug: after the win, NEXT and the ads, the game showed the finished L187 board again (empty
grid, 0/5, no sand: frames 00272, 00305). Each such frame started a "level" that ended at once
as an inferred win, so the run counted 3 wins and stopped "--levels 3" on the menu instead of
pressing PLAY for L188.

tests/data/run-20261021-000105/: 00260 (the level's end, still in play), 00268 (win screen),
00271 (ad "take a break"), 00272 (the finished board). The run's menu frame 00308 has coins
flying over PLAY (read "other"), so the test ends on the dataset's home map.
"""
import json
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

from runlog import load_events  # noqa: E402
from tests.test_bot import S023, Tapper, make_bot, run, screen_taps, shot  # noqa: E402
from tests.test_bot import analyzer, cfg, cfg6  # noqa: E402,F401  (fixtures)

DATA = Path(__file__).resolve().parent / "data" / "run-20261021-000105"


def frame(name):
    img = cv2.imread(str(DATA / f"{name}.png"))
    assert img is not None, name
    return img


def test_finished_board_is_not_a_new_level(cfg6, analyzer, tmp_path):  # noqa: F811
    play, win, ad = frame("00260-wait"), frame("00268-screen-tap-next"), frame("00271-screen-ad-wait")
    empty, menu = frame("00272-wait"), shot(S023)   # 00308: coins fly over PLAY ("other")
    frames = [play] * 2 + [win] * 8 + [ad] * 2 + [empty] * 3 + [ad] * 2 + [empty] * 3 + [menu] * 6
    bot, log = make_bot(cfg6, analyzer, tmp_path, frames, tapper=Tapper(), levels=3)
    run(bot, log)
    events = load_events(log.dir)
    assert [e["level"] for e in events if e["type"] == "level_start"] == [1]
    assert [e["result"] for e in events if e["type"] == "level_end"] == ["win"]
    assert ("level_complete", "next") in screen_taps(log)
    assert screen_taps(log)[-1] == ("menu", "play")           # L188, not "3 levels played"
    assert any(e["type"] == "finished_board" for e in events)
    res = json.loads((log.dir / "result.json").read_text())
    assert res["levels"] == 1 and res["wins"] == 1


def test_the_finished_board_reads_empty(analyzer):  # noqa: F811
    from cells import Kind

    res = analyzer.analyze(frame("00272-wait"), log=lambda s: None, now=0.0)
    assert res.screen == "gameplay" and res.state is not None
    assert all(c.kind is Kind.EMPTY for c in res.state.cells())
    assert sum(res.state.sand.values()) < 0.01
    assert res.speed_read is True                              # the 2x button was still green
