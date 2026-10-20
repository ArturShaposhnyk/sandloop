"""How fast belt buckets really fill, from live runs (calibrates the simulator's belt speed).

    python -m tools.belt_speed                         # the three L187 runs of 2026-10-20
    python -m tools.belt_speed debug/runs/<stamp> ...  # these runs

Per run: belt.csv (the belt per analysed frame) + events.jsonl (the contact-layer share of every
colour in that frame). A sample = one colour with exactly one unfinished bucket read in two
frames at most ``MAX_DT`` apart: fill gain per second vs the colour's contact-layer share (the
sand the belt can really reach). Prints the median / mean rate per share band, the fit
``rate = fill_rate x (share - fill_min_exposed)`` capped at ``fill_max`` (planner: in
config.yaml), the bot's seconds per tap (``tap_s``) and, from levels.jsonl, how the
level plans' predicted belt compared with the real one ("plan check" records).

Belt speed 1x and 2x are calibrated separately (belt.csv column ``speed``, from the ▶▶2x
button; runs before it have no column and were all played at 1x): the 1x fit goes to
``planner.fill_*``, the 2x fit to ``planner.speed_x2_fill_*``. Frames with an unread speed
are left out.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

RUNS = ["debug/runs/20261020-142035", "debug/runs/20261020-145310",
        "debug/runs/20261020-161536"]
MAX_DT = 2.5


def parse_slots(text: str) -> Dict[str, List[Optional[int]]]:
    """belt.csv slots "orange 45|orange/cream ?" -> colour (collecting stage) -> fills."""
    out: Dict[str, List[Optional[int]]] = defaultdict(list)
    for part in filter(None, text.split("|")):
        name, _, fill = part.rpartition(" ")
        out[name.split("/")[0]].append(None if fill == "?" else int(fill))
    return out


def row_speed(row: Dict[str, str]) -> Optional[int]:
    """Belt speed of a belt.csv row: 1 / 2, None = not read; no column (older runs) = 1x."""
    if "speed" not in row or row["speed"] is None:
        return 1
    return int(row["speed"]) if row["speed"] else None


def samples(run: Path, speed: int = 1) -> List[Tuple[float, float, str]]:
    """(rate %/s, contact share, colour) of one run, rows at belt speed ``speed`` only."""
    shares = {}
    for line in open(run / "events.jsonl"):
        e = json.loads(line)
        if e.get("type") == "analysis":
            shares[e["frame"]] = e.get("exposed") or {}
    rows = list(csv.DictReader(open(run / "belt.csv")))
    out = []
    for a, b in zip(rows, rows[1:]):
        dt = float(b["t"]) - float(a["t"])
        if not 0 < dt <= MAX_DT or int(a["frame"]) not in shares:
            continue
        if row_speed(a) != speed or row_speed(b) != speed:
            continue
        sa, sb = parse_slots(a["slots"]), parse_slots(b["slots"])
        for colour, fa in sa.items():
            fb = sb.get(colour, [])
            if len(fa) != 1 or len(fb) != 1 or None in fa + fb or max(fa[0], fb[0]) >= 100:
                continue
            out.append(((fb[0] - fa[0]) / dt, shares[int(a["frame"])].get(colour, 0.0), colour))
    return out


def fit(points: Sequence[Tuple[float, float, str]]) -> Tuple[float, float, float]:
    """(fill_rate, fill_min_exposed, fill_max) with the least squared error (grid search)."""
    best = (float("inf"), 0.0, 0.0, 0.0)
    for e0 in [i / 100 for i in range(31)]:
        for cap in range(10, 65, 5):
            for a in range(20, 305, 5):
                err = sum((min(cap, a * max(0.0, x - e0)) - r) ** 2 for r, x, _ in points)
                best = min(best, (err, float(a), e0, float(cap)))
    return best[1], best[2], best[3]


def tap_seconds(runs: Iterable[Path]) -> Optional[float]:
    """Seconds between two taps of the bot when it does not wait (lower quartile of the gaps;
    taps of one burst count once)."""
    gaps = []
    for run in runs:
        ts = [e["t"] for e in map(json.loads, open(run / "events.jsonl"))
              if e.get("type") == "tap" and e.get("burst", 1) == 1]
        gaps += [b - a for a, b in zip(ts, ts[1:]) if b - a < 15]
    return round(sorted(gaps)[len(gaps) // 4], 1) if gaps else None


def plan_checks(levels: Path) -> List[str]:
    """Predicted vs real belt per plan step (levels.jsonl "plans" -> "checks")."""
    lines = []
    if not levels.exists():
        return lines
    for line in open(levels):
        rec = json.loads(line)
        for plan in rec.get("plans", []):
            for c in plan.get("checks", []):
                lines.append(f"  {rec.get('run', '?')} plan @{plan['t']:.0f}s step {c['step']}: "
                             f"t {c['t']:.0f}s (plan {c['t_plan']:.0f}s), k {c['k']} (plan "
                             f"{c['k_plan']}), belt {c['belt']} (plan {c['belt_plan']})"
                             + (f" - gave up: {c['gave_up']}" if c.get("gave_up") else ""))
    return lines


def report(speed: int, points: Sequence[Tuple[float, float, str]], runs: int, keys: str) -> None:
    print(f"belt speed {speed}x: {len(points)} samples from {runs} run(s)")
    bands: Dict[float, List[float]] = defaultdict(list)
    for r, x, _ in points:
        bands[min(0.5, int(x * 20) / 20)].append(r)
    for b in sorted(bands):
        v = bands[b]
        print(f"  contact share {b:.2f}-{b + 0.05:.2f}: {len(v):3d} samples, median "
              f"{statistics.median(v):5.1f} %/s, mean {statistics.mean(v):5.1f} %/s")
    if points:
        rate, e0, cap = fit(points)
        print(f"  fit: rate {rate}  min_exposed {e0:.2f}  max {cap}  (config.yaml planner: "
              f"{keys})")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="*", default=RUNS)
    ap.add_argument("--levels", default="debug/levels.jsonl")
    args = ap.parse_args(argv)
    runs = [Path(r) for r in args.runs if (Path(r) / "belt.csv").exists()]
    for speed, keys in ((1, "fill_rate / fill_min_exposed / fill_max"),
                        (2, "speed_x2_fill_rate / speed_x2_fill_min_exposed / speed_x2_fill_max")):
        report(speed, [p for r in runs for p in samples(r, speed)], len(runs), keys)
    print(f"tap_s (s per tap when the bot does not wait): {tap_seconds(runs)}")
    checks = plan_checks(Path(args.levels))
    if checks:
        print("plan checks (predicted vs real belt):")
        print("\n".join(checks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
