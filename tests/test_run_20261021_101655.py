"""Regressions from the live run debug/runs/20261021-101655 (L189, lost 3 times; the flow between
the attempts - Quit, ✕, Try Again, ad ✕ - worked).

- The revive popup sliding in (frame 00507: "Keep Playing" matched 0.49) was read as gameplay:
  the board under it read "green", and that read became the level's end record (and the
  level memory's capacity correction). Now screen "popup" (``screens.popup_bar``).
- Level memory did not make the attempts differ: the plans were never stored (the analyzer
  was reset before the record), attempt 3 got its own signature (its first frame read the top
  row face-down), and bans compared whole-level sequences with a re-plan's own taps.
- The user's player hints: remember colours seen under locks (r3c1 cyan under the ice), value
  progress / opened cells, punish idle belt slots, burst back-to-back plan steps.

tests/data/run-20261021-101655/: 00005 (L189 start, 2x), 00507 (popup sliding in), levels.jsonl
(the 3 attempts, trimmed).
"""
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

cv2 = pytest.importorskip("cv2")

from levelmemory import LevelMemory, _record_signature, level_signature  # noqa: E402
from levelplan import PlanRunner, plan_level  # noqa: E402
from sim import Picture, Sim, SimConfig  # noqa: E402
from tests.test_state import analyzer, make_state  # noqa: E402,F401  (fixture)

DATA = Path(__file__).resolve().parent / "data" / "run-20261021-101655"


def frame(name):
    img = cv2.imread(str(DATA / f"{name}.png"))
    assert img is not None, name
    return img


def test_popup_sliding_in_is_no_gameplay_frame(analyzer):  # noqa: F811
    res = analyzer.analyze(frame("00507-wait"), log=lambda s: None, now=0.0)
    assert res.screen == "popup" and res.state is None
    assert analyzer.analyze(frame("00005-tap"), log=lambda s: None, now=0.0).screen == "gameplay"


def test_signature_ignores_the_faces():
    up = ["cyan-up cream-up x", "ice15 red-down ."]
    assert level_signature(up) == level_signature(["cyan-down cream-down x", "ice15 red-up ."])
    assert level_signature(up) != level_signature(["cyan-up cream-up x", "ice14 red-down ."])


def test_memory_of_the_three_attempts():
    mem = LevelMemory(DATA / "levels.jsonl")
    sigs = {_record_signature(r) for r in mem.records}
    assert len(sigs) == 1                                   # attempt 3 is the same board now
    sig = sigs.pop()
    assert len(mem.banned(sig)) == 3
    assert mem.revealed(sig) == {(3, 1): ("cyan", None)}    # attempt 1's end: the ice opened
    caps = mem.capacity_shares(sig)                         # attempts 2 / 3 ended on the popup
    assert caps["cyan"] == pytest.approx(0.0335, abs=1e-3)


def test_sim_opens_the_ice_to_the_remembered_colour():
    st = make_state(["a ice1"], k=0)
    pic = Picture(["a", "b"], [[1, 0, 0]] * 4, contact=1)
    s = Sim.start(st, pic, SimConfig(slots=5), hidden={(0, 1): ("b", None)})
    s = s.tap((0, 0))
    assert s.opened == 1 and s.grid.cell((0, 1)).color == "b" and (0, 1) in s.moves()
    blind = Sim.start(st, pic, SimConfig(slots=5)).tap((0, 0))
    assert blind.grid.cell((0, 1)).color is None and blind.unknown_left()


def test_sim_counts_idle_slots_and_fast_bursts():
    st = make_state(["a b c"], k=0)
    pic = Picture(["a", "b", "c"], [[0, 0, 1, 2]] * 4, contact=1)   # a at the bottom
    sc = SimConfig(slots=5, tap_s=4.0, burst_tap_s=1.0, burst_max=3)
    s = Sim.start(st, pic, sc).tap((0, 1))          # b is buried: idle while it waits
    assert s.idle == pytest.approx(4.0) and s.t == pytest.approx(4.0)
    s = s.tap((0, 2)).tap((0, 0))                    # the burst: 1 s per follow-up tap
    assert s.t == pytest.approx(6.0)
    slow = Sim.start(st, pic, dataclasses.replace(sc, burst_max=1))
    assert slow.tap((0, 1)).tap((0, 2)).t == pytest.approx(8.0)


def test_plan_bans_whole_level_sequences():
    from config import PlannerConfig

    st = make_state(["x b"], k=0)
    pic = Picture(["b"], [[0, 0]] * 4, contact=1)
    pc = PlannerConfig()
    assert [s.pos for s in plan_level(st, pic, pc).steps] == [(0, 1)]
    banned = {((0, 0), (0, 1))}                      # the attempt that lost: r0c0, then r0c1
    assert plan_level(st, pic, pc, banned, done=[(0, 0)]).steps == []
    assert [s.pos for s in plan_level(st, pic, pc, banned).steps] == [(0, 1)]


def test_start_plan_bursts_back_to_back_steps(analyzer, tmp_path):  # noqa: F811
    from config import load_config

    cfg = load_config()
    res = analyzer.analyze(frame("00005-tap"), log=lambda s: None, now=0.0)
    runner = PlanRunner(cfg, tmp_path / "levels.jsonl")
    runner.x2 = True
    d = runner.decide(frame("00005-tap"), res.view, res.state, res.decision, 0.0,
                      log=lambda s: None)
    assert d.move is not None and d.move.level_plan == f"1/{len(runner.plan.steps)}"
    assert 1 <= len(d.move.follow) <= cfg.planner.burst_max - 1
    assert all(f.pos in res.state.tappable for f in d.move.follow)
    assert [f.pos for f in d.move.follow] == [s.pos for s in runner.plan.steps[1:1 + len(
        d.move.follow)]]


def test_watch_hidden_records_the_colour_under_the_ice(analyzer):  # noqa: F811
    analyzer.reset_level(track=False)
    analyzer._watch_hidden(make_state(["red ice2"]), lambda s: None)
    opened = make_state(["red cyan"])
    analyzer._watch_hidden(opened, lambda s: None)
    assert analyzer.revealed == {}                   # once is not enough (a fading lock)
    analyzer._watch_hidden(opened, lambda s: None)
    assert analyzer.revealed == {"0,1": "cyan"}


def test_level_record_is_written_before_the_reset():
    """The plans of a lost level reached levels.jsonl as [] (run 101655): the analyzer (and its
    PlanRunner) was reset first."""
    from levels import LevelMixin

    seen = {}

    class An:
        planner = SimpleNamespace(log=[{"why": "start"}])

        def reset_level(self):
            self.planner = SimpleNamespace(log=[])

    class Log:
        def frame(self, *a):
            return "x.png"

        def log(self, *a):
            pass

        def event(self, *a, **k):
            pass

        def t(self):
            return 0.0

        def level(self, **k):
            seen.update(k)

    bot = SimpleNamespace(last_sand=0.3, bc=SimpleNamespace(win_sand_max=0.01), log=Log(),
                          level_no=1, frame_no=1, analyzer=An(), taps=1, level_sig="s",
                          level_t0=0.0, level_taps=[], level_first=None, level_last=None,
                          session=SimpleNamespace(record=lambda r: None), in_level=True)
    bot._record_level = lambda result, how: LevelMixin._record_level(bot, result, how)
    LevelMixin._level_left(bot, SimpleNamespace(screen="revive"), None)
    assert seen["result"] == "fail" and seen["plans"] == [{"why": "start"}]
