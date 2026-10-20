"""Belt speed 2x (user, 2026-10-20: the ▶▶2x button under the k/5 counter is free and resets
every level): hud.read_speed / SpeedTracker, bot.py taps it at level start while it reads off,
the planner's fill speeds follow it (levelplan.sim_config), tools.belt_speed fits 1x / 2x apart."""
import dataclasses
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

from config import load_config  # noqa: E402
from hud import SpeedTracker, read_speed  # noqa: E402
from levelplan import Plan, PlanRunner, sim_config  # noqa: E402
from runlog import load_events  # noqa: E402
from tests.test_bot import S039, S056, Tapper, make_bot, run, shot  # noqa: E402
from tests.test_bot import analyzer  # noqa: E402,F401  (fixture)
from tests.test_bot import cfg as bot_cfg  # noqa: E402,F401  (fixture)

ROOT = Path(__file__).resolve().parent
OFF = ROOT / "data" / "run-20261020-224631" / "00228-tap.png"       # grey: 1x (live run)
CARD = ROOT / "data" / "run-20261020-224631" / "00427-screen-other-wait.png"


@pytest.fixture(scope="module")
def cfg(bot_cfg):  # noqa: F811
    """test_bot.py's config with the ▶▶2x tap on (it is off there)."""
    return dataclasses.replace(bot_cfg, bot=dataclasses.replace(bot_cfg.bot, speed_x2=True))


def test_read_speed_green_on_grey_off_else_none():
    bc = load_config().bot
    assert read_speed(shot(S056), bc) is True                       # L183 live, 2x on
    assert read_speed(cv2.imread(str(OFF)), bc) is False
    assert read_speed(cv2.imread(str(CARD)), bc) is None            # ad end card: no button


def test_speed_tracker_needs_equal_readings():
    t = SpeedTracker(2)
    assert not t.feed(False) and t.state is None
    assert not t.feed(None) and t.state is None                     # unread frames are skipped
    assert t.feed(False) and t.state is False
    assert not t.feed(True) and t.state is False
    assert t.feed(True) and t.state is True


def test_sim_config_scales_fill_speed_not_tap_time():
    pc = load_config().planner
    one, two = sim_config(pc, 5), sim_config(pc, 5, x2=True)
    assert two.fill_rate == one.fill_rate * pc.speed_x2_factor
    assert two.fill_max == one.fill_max * pc.speed_x2_factor
    assert (two.tap_s, two.fill_min_exposed) == (one.tap_s, one.fill_min_exposed)
    pc = dataclasses.replace(pc, speed_x2_fill_rate=150.0, speed_x2_fill_max=45.0,
                             speed_x2_fill_min_exposed=0.05)          # calibrated at 2x
    two = sim_config(pc, 5, x2=True)
    assert (two.fill_rate, two.fill_max, two.fill_min_exposed) == (150.0, 45.0, 0.05)


def test_speed_change_replans():
    runner = PlanRunner(load_config())
    runner.plan, runner.last_plan_t = Plan([], "open", 0, 0, 0), 5.0
    logs = []
    runner.set_speed(True, logs.append)
    assert runner.x2 and runner.plan is None and runner.last_plan_t < 0
    assert logs == ["level plan: belt speed 2x - re-planning"]
    runner.set_speed(True, logs.append)
    assert len(logs) == 1


def test_bot_taps_the_speed_button_when_it_reads_off(cfg, analyzer, tmp_path):  # noqa: F811
    off, fail = cv2.imread(str(OFF)), shot(S039)
    tapper = Tapper()
    cfg = dataclasses.replace(cfg, bot=dataclasses.replace(cfg.bot, speed_retap_s=0.0))
    bot, log = make_bot(cfg, analyzer, tmp_path, [off] * 12 + [fail] * 3, tapper=tapper)
    run(bot, log)
    z = cfg.bot.speed_zone
    centre = ((z[0] + z[2]) / 2, (z[1] + z[3]) / 2)
    speed = [e for e in load_events(log.dir) if e["type"] == "speed_tap"]
    assert len(speed) == cfg.bot.speed_max_taps                     # it stayed grey: 2, then 1x
    assert tapper.taps[:1] == [centre]
    assert sum(1 for t in tapper.taps if t == centre) == cfg.bot.speed_max_taps
    text = (log.dir / "bot.log").read_text()
    assert "belt speed 2x: off" in text and "still off after 2 tap(s)" in text


def test_bot_leaves_the_button_alone_when_on_or_disabled(cfg, analyzer, tmp_path):  # noqa: F811
    on, fail = shot(S056), shot(S039)
    bot, log = make_bot(cfg, analyzer, tmp_path / "on", [on, on, fail, fail, fail])
    run(bot, log)
    assert not [e for e in load_events(log.dir) if e["type"] == "speed_tap"]
    assert "belt speed 2x: on" in (log.dir / "bot.log").read_text()
    c = dataclasses.replace(cfg, bot=dataclasses.replace(cfg.bot, speed_x2=False))
    off = cv2.imread(str(OFF))
    bot, log = make_bot(c, analyzer, tmp_path / "off", [off] * 4 + [fail] * 3)
    run(bot, log)
    assert not [e for e in load_events(log.dir) if e["type"] == "speed_tap"]


def test_belt_speed_tool_splits_by_speed(tmp_path):
    from tools.belt_speed import row_speed, samples

    assert row_speed({"t": "1"}) == 1                                # old runs: no column = 1x
    assert row_speed({"speed": "2"}) == 2 and row_speed({"speed": ""}) is None
    run = tmp_path / "run"
    run.mkdir()
    (run / "events.jsonl").write_text(
        "\n".join('{"type": "analysis", "frame": %d, "exposed": {"orange": 0.3}}' % i
                  for i in range(1, 5)) + "\n")
    (run / "belt.csv").write_text("t,frame,k,slots,mean_fill,speed\n"
                                  "0,1,1,orange 10,10,1\n1,2,1,orange 30,30,1\n"
                                  "2,3,1,orange 30,30,2\n3,4,1,orange 70,70,2\n")
    assert [round(r) for r, _, _ in samples(run, 1)] == [20]
    assert [round(r) for r, _, _ in samples(run, 2)] == [40]
