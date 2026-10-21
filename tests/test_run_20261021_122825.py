"""Run debug/runs/20261021-122825 (L190): the bot idled 116 -> 251 s and 271 -> 396 s with "no
bucket passes the rules" while no belt bucket gained sand. The belt reader dropped one of the
buckets every other frame (darkgreen 93 / an orange, or read the orange/cream tower with "?"):
BeltTracker followed the lower count after ``forget_s``, the bucket "arrived" again and the
stall clock restarted every ~15 s; ``belt_idle`` counted k > buckets read as collecting.
Fixed with the k/5 counter (state.BeltTracker) and ``LevelState.belt_tracked``.
"""
import csv
from pathlib import Path
from types import SimpleNamespace

from config import load_config
from endgame import belt_stuck
from state import BeltTracker, ConveyorBucket
from strategy import belt_idle

DATA = Path(__file__).resolve().parent / "data" / "run-20261021-122825"


def _slots(text):
    out = []
    for part in filter(None, text.split("|")):
        name, fill = part.rsplit(" ", 1)
        cols = name.split("/")
        out.append(ConveyorBucket(cols[0], None if fill == "?" else int(fill),
                                  stages=tuple(cols[1:])))
    return out


def _replay():
    sc = load_config().strategy
    t = BeltTracker(sc.stall_min_growth, sc.stall_forget_s)
    for row in csv.DictReader((DATA / "belt.csv").open()):
        k = int(row["k"]) if row["k"] else None
        conv = _slots(row["slots"])
        stalled = t.update(conv, float(row["t"]), k)
        st = SimpleNamespace(conveyor=conv, conveyor_k=k, belt_stalled=stalled,
                             belt_tracked=t.tracked)
        yield float(row["t"]), st, sc


def test_flickering_belt_counts_as_stuck():
    idle_at = stuck_at = None
    for now, st, sc in _replay():
        if 117 <= now <= 250:
            if idle_at is None and belt_idle(st, sc):
                idle_at = now
            if idle_at is not None:
                # once stuck, the belt stays stuck: no reread restarts the clock
                s = belt_stuck(st, sc)
                if stuck_at is None and s is not None and s >= sc.lost_stuck_s:
                    stuck_at = now
                if stuck_at is not None:
                    assert s is not None and s >= sc.lost_stuck_s, now
    assert idle_at is not None and idle_at < 125     # was never steady before 251 s
    assert stuck_at is not None and stuck_at < 140


def test_two_oranges_read_one_at_a_time_stall():
    """271 -> 300 s: k 2, "orange 57|orange 0" read one or both: stuck within lost_stuck_s."""
    out = None
    for now, st, sc in _replay():
        if now >= 271.8:
            out = (now, belt_idle(st, sc), belt_stuck(st, sc))
    now, idle, stuck = out
    assert idle and stuck is not None and stuck > 25.0
