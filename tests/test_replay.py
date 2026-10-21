"""tools/replay.py: the comparisons of the replay harness (synthetic, no frames)."""
from pathlib import Path

from cells import Cell, Kind
from config import load_config
from sim import Picture
from state import ConveyorBucket
from tests.test_state import make_state
from tools.replay import Start, Traj, pair_fills, real_gain, replay, summarize


def test_pair_fills_by_colour_high_first():
    real = [("red", 40), ("red", 10), ("blue", 100), ("cyan", None), ("blue", 55)]
    sim = [("red", 12), ("red", 45), ("blue", 50)]
    assert sorted(pair_fills(real, sim)) == [2.0, 5.0, 5.0]   # red 40/45, 10/12; blue 55/50
    # one of two same-colour buckets read: paired with the nearer one (L190 orange 0 / 57)
    assert pair_fills([("orange", 0)], [("orange", 57), ("orange", 2)]) == [2.0]


def test_real_gain_ignores_two_buckets_read_alternately():
    assert not real_gain([0, 57], [57, 0, 57, 58])       # L190: orange 57 / 0 flicker, jitter
    assert real_gain([0, 57], [57, 4])                    # the 0 % one grew
    assert real_gain([93], [93, 97])
    assert not real_gain([93], [93, 94])                  # < 3 points: label jitter


def test_summarize_counts_optimistic_stalls():
    t = Traj(Start("r", 1, 0.0, Path("x"), False))
    t.gains = [("red", True, True), ("blue", True, False), ("cyan", False, False)]
    t.fills = [(6.0, 10.0), (12.0, 20.0)]
    t.k = [(6.0, 0), (6.5, 1)]
    s = summarize([t])
    assert s["gain"] == {"tp": 1, "fp": 1, "fn": 0, "tn": 1, "accuracy": 0.667,
                         "gain_recall": 1.0, "gain_precision": 0.5, "stall_recall": 0.5}
    assert s["fill_mae"]["5-10s"] == {"n": 1, "mae": 10.0}
    assert s["k_exact"]["5-10s"] == 0.5


def _events():
    conv = [{"color": "red", "fill": 0}]
    ev = [{"type": "analysis", "t": 0.0, "frame": 1, "screen": "gameplay", "k": 1,
           "conveyor": conv, "sand": {"red": 0.5, "blue": 0.5},
           "exposed": {"red": 0.5, "blue": 0.5}}]
    for i in range(1, 13):
        ev.append({"type": "analysis", "t": float(i), "frame": 1 + i, "screen": "gameplay",
                   "k": 1, "conveyor": [{"color": "red", "fill": min(100, 9 * i)}],
                   "sand": {"red": 0.5 - 0.02 * i, "blue": 0.5},
                   "exposed": {"red": 0.5, "blue": 0.5}})
    return ev


def test_replay_follows_the_real_belt():
    """A red bucket at 0 % with red exposed: the simulator fills it, the real one did too."""
    cfg = load_config()
    state = make_state(["red"], k=1, conveyor=[ConveyorBucket("red", 0)])
    pic = Picture(["red", "blue"], [[0, 1] * 10 for _ in range(20)], contact=2)
    tr = replay(Start("r", 1, 0.0, Path("x"), False), _events(), state, pic, cfg.planner, 30.0)
    assert tr.end == "horizon"
    assert tr.gains == [("red", True, True)]
    assert tr.fills and tr.k[0] == (0.0, 0)


def test_replay_stops_at_a_tap_it_cannot_follow():
    cfg = load_config()
    state = make_state(["red"], k=1, conveyor=[ConveyorBucket("red", 0)])
    state.grid[0][0] = Cell(0, 0, Kind.UNKNOWN)
    ev = _events()
    ev.insert(3, {"type": "tap", "t": 2.5, "pos": [0, 0], "executed": True})
    pic = Picture(["red", "blue"], [[0, 1] * 10 for _ in range(20)], contact=2)
    tr = replay(Start("r", 1, 0.0, Path("x"), False), ev, state, pic, cfg.planner, 30.0)
    assert tr.end == "unknown tap" and tr.gains == []


def test_crops_and_pruning(tmp_path):
    """bot.crop_every_s: picture + belt rows saved under crops/; above crops_max_mb the oldest
    runs' crops/ go (never the running one), their other files stay."""
    import numpy as np

    from runlog import RunLog, prune_crops

    cfg = load_config()
    clk = iter(range(100)).__next__
    old = RunLog(tmp_path, cfg, {}, clock=lambda: 0.0, stamp="20260101-000000")
    img = (np.random.default_rng(0).random((1000, 400, 3)) * 255).astype(np.uint8)
    assert old.crop(1, img, (0.1, 0.5)) == "crops/00001.png"
    old.close()
    new = RunLog(tmp_path, cfg, {}, clock=lambda: float(clk()), stamp="20260102-000000")
    new.crop(7, img, (0.1, 0.5))
    rows = (new.dir / "crops" / "index.csv").read_text().splitlines()
    assert rows[1].split(",")[1:] == ["7", "00007.png", "100", "500", "1000", "400"]
    assert prune_crops(tmp_path, 0.0001, keep=new.dir, log=lambda s: None) == 1
    assert not (old.dir / "crops").exists() and (old.dir / "events.jsonl").exists()
    assert (new.dir / "crops" / "00007.png").exists()
    new.close()
