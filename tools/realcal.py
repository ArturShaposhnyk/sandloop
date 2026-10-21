"""Phase D: does the simulator reproduce the real game? (user, 2026-10-21: the real game is the
judge; self-play only after the simulator reproduces real outcomes.)

Two checks, both from the recorded runs:

* **Bucket episodes** - every bucket the bot tapped is followed on the real belt by the belief
  (belief.py: slot identity, fills only grow, the k counter decides when one left) - its fills
  and when it left the belt (or that it was still there / stuck). The simulator starts on the
  frame of that tap (the frame's picture and belt, tools/replay.py's cached reads) with the
  run's real taps at their real times and follows the same bucket. Compared: did it leave
  within ``--horizon`` (both / only real / only sim / neither), the error of its departure
  time, and of its fill at the real readings.
* **Outcomes** - the real tap sequence of a level, replayed open loop from its start frame at
  the real times: the won attempts (L187 000105, L189 112241 #2) must win, the lost ones lose.
  A tap the simulated board cannot take (belt full, cell not a bucket) counts as blocked.

    python -m tools.realcal                         # all runs, today's simulator
    python -m tools.realcal --world contact=1.5 --world fill_rate=120

``--world``: SimConfig fields (fill_rate, fill_min_exposed, fill_max) and ``contact`` (x the
picture's contact layer), applied to the 2x belt (all levels compared are 2x).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# level attempts with a known outcome: (level, run, attempt in levels.jsonl, real result)
OUTCOMES = [("L187", "20261021-000105", 1, "win"), ("L189", "20261021-112241", 2, "win"),
            ("L187", "20261020-224631", 1, "fail"), ("L187", "20261020-171133", 1, "fail"),
            ("L189", "20261021-112241", 1, "fail"), ("L189", "20261021-101655", 3, "fail"),
            ("L190", "20261021-143729", 1, "fail"), ("L190", "20261021-232340", 1, "fail"),
            ("L190", "20261021-232340", 2, "fail"), ("L190", "20261021-232340", 3, "fail")]


@dataclass
class Episode:
    run: str
    colour: str
    frame: int                       # the tap's frame (the simulator starts there)
    t: float                         # tap time
    share: float                     # its colour's contact-layer share at the tap (vision)
    fills: List[Tuple[float, int]] = field(default_factory=list)   # (s after the tap, fill)
    left: Optional[float] = None     # s after the tap it left the belt
    seen_until: float = 0.0          # s after the tap the level was still observed


def real_episodes(cfg, run_dir: Path, events: List[Dict[str, Any]]) -> List[Episode]:
    from belief import BeltBelief
    from state import ConveyorBucket

    dc = dataclasses.replace(cfg.decider, rate_window_s=1e9)   # keep every reading
    out: List[Episode] = []
    mine: Dict[int, Tuple[Any, Episode]] = {}
    pairs: List[Tuple[Any, Episode]] = []
    b = BeltBelief(dc)
    exposed: Dict[str, float] = {}

    def close(belief, now: float) -> None:
        for slot, when in belief.left:
            if id(slot) in mine and mine[id(slot)][1].left is None:
                mine[id(slot)][1].left = when - mine[id(slot)][1].t
        belief.left.clear()
        for slot, ep in mine.values():
            ep.seen_until = max(ep.seen_until, now - ep.t)

    last_t = 0.0
    for e in events:
        if e["type"] in ("level_start", "level_end"):
            close(b, last_t)
            mine.clear()
            b = BeltBelief(dc)
            continue
        if e["type"] == "tap" and e.get("executed", True):
            b.tapped(e["color"], tuple(e.get("stages") or ()), e["t"])
            slot = b.slots[-1]
            ep = Episode(run_dir.name, e["color"], e["frame"], e["t"],
                         exposed.get(e["color"], 0.0))
            mine[id(slot)] = (slot, ep)
            pairs.append((slot, ep))
            out.append(ep)
            continue
        if (e["type"] != "analysis" or e.get("screen") != "gameplay" or e.get("error")
                or e.get("k") is None):
            continue
        exposed = e.get("exposed") or {}
        conv = [] if e["k"] == 0 else [
            ConveyorBucket(x.get("color"), x.get("fill"), False, tuple(x.get("stages") or ()))
            for x in e.get("conveyor") or []]
        b.update(conv, e["k"], e["t"])
        last_t = e["t"]
        close(b, e["t"])
    close(b, last_t)
    for slot, ep in pairs:
        ep.fills = [(t - ep.t, f) for t, f in slot.hist if slot.stage == 0]
    return out


def world_config(cfg, slots: int, world: Dict[str, float]):
    from levelplan import sim_config

    world = dict(world)
    contact = world.pop("contact", 1.0)
    return dataclasses.replace(sim_config(cfg.planner, slots, True), **world), contact


def sim_episode(cfg, ep: Episode, events, state, pic, world, horizon: float):
    """(left after s or None, [(s, simulated fill) at the real readings]) - None if the
    simulator cannot follow (the tapped cell is no bucket there)."""
    from cells import Kind
    from sim import Sim

    sc, contact = world_config(cfg, state.conveyor_slots, world)
    pic = pic.copy()
    pic.contact = max(1, round(pic.contact * contact))
    pic._layer = None
    s = Sim.start(state, pic, sc)
    s.grid.disagreements = []
    s.grid.tappable = s.grid.playable()
    start_t = next((e["t"] for e in events if e["type"] == "analysis"
                    and e.get("frame") == ep.frame), ep.t)
    taps = [e for e in events if e["type"] == "tap" and e.get("executed", True)
            and start_t <= e["t"] <= ep.t + horizon]
    slot, now, fills = None, start_t, []
    want = sorted(t for t, _ in ep.fills)
    for e in taps + [{"t": ep.t + horizon, "end": True}]:
        while now < e["t"]:
            s.lap()
            now += s.sc.lap_s
            if slot is not None:
                dt = now - ep.t
                if not any(x is slot for x in s.belt):
                    return dt, fills
                while want and want[0] <= dt:
                    cap = max(1, s.capacity.get(slot.colour, 1))
                    fills.append((want.pop(0), int(100 * (1 - slot.left / cap))))
        if e.get("end"):
            break
        pos = tuple(e["pos"])
        cell = s.grid.cell(pos) if 0 <= pos[0] < s.grid.rows and 0 <= pos[1] < s.grid.cols else None
        if cell is None or cell.kind is not Kind.BUCKET or not cell.color or s.free <= 0:
            if abs(e["t"] - ep.t) < 1e-6:
                return None
            continue
        s.place(pos)
        if abs(e["t"] - ep.t) < 1e-6:
            slot = s.belt[-1]
    return None, fills


def outcome(cfg, level: str, run: str, attempt: int, world, starts) -> Dict[str, Any]:
    """The real tap sequence of an attempt, open loop from the level's start frame."""
    from cells import Kind
    import tools.selfplay as sp

    rec = next(json.loads(line) for line in
               (Path(cfg.paths.debug) / "levels.jsonl").read_text().splitlines()
               if json.loads(line)["run"] == run and json.loads(line)["level"] == attempt)
    taps = [(t["t"], tuple(t["pos"])) for t in rec["tap_log"] if t.get("executed", True)]
    state, pic = starts[level]
    w = sp.make_world(cfg, state, pic, 0, 0.0, sp.truth(cfg, level), world)
    total, now, blocked, last_gain = w.pic.total(), 0.0, 0, 0.0
    for t, pos in taps:
        while now < t - taps[0][0]:
            if w.lap() > 0:
                last_gain = now
            now += w.sc.lap_s
        cell = w.grid.cell(pos)
        if (cell is None or cell.kind is not Kind.BUCKET or not cell.color or w.free <= 0
                or pos not in w.grid.playable()):
            blocked += 1
            continue
        w.place(pos)
    for _ in range(300):
        if w.lap() > 0:
            last_gain = now
        now += w.sc.lap_s
    won = w.won or w.pic.total() <= 0.005 * total
    return {"level": level, "run": run, "attempt": attempt, "real": rec["result"],
            "sim": "win" if won else "fail", "blocked": blocked, "taps": len(taps),
            "collected": round(w.collected / max(1, total), 3),
            "sim_end_s": round(last_gain), "real_end_s": round(rec.get("duration_s") or 0)}


def evaluate(cfg, run_dirs: Sequence[Path], world: Dict[str, float], horizon: float = 40.0,
             log=print) -> Dict[str, Any]:
    from runlog import load_events
    from tools.replay import Reader, starts_of
    import tools.selfplay as sp

    reader = Reader(cfg, Path(cfg.paths.debug) / "replay" / "cache", cell_px=cfg.planner.cell_px)
    conf = defaultdict(int)          # (real left in horizon, sim left in horizon) -> count
    life_err, fill_err, skipped = [], [], 0
    per_share: Dict[str, List[float]] = defaultdict(list)
    for rd in run_dirs:
        if not (rd / "events.jsonl").exists():
            continue
        events = load_events(rd)
        starts = {s.frame: s for s in starts_of(rd, events)}
        for ep in real_episodes(cfg, rd, events):
            st = starts.get(ep.frame)
            if st is None or (ep.left is None and ep.seen_until < horizon):
                skipped += 1          # no frame, or the level ended before we know
                continue
            state, pic = reader.read(st)
            if state is None or pic is None:
                skipped += 1
                continue
            got = sim_episode(cfg, ep, events, state, pic, world, horizon)
            if got is None:
                skipped += 1
                continue
            sim_left, fills = got
            real_in = ep.left is not None and ep.left <= horizon
            sim_in = sim_left is not None
            conf[(real_in, sim_in)] += 1
            if real_in and sim_in:
                life_err.append(sim_left - ep.left)
                band = "<10%" if ep.share < 0.1 else "10-25%" if ep.share < 0.25 else ">25%"
                per_share[band].append(sim_left - ep.left)
            real = dict(ep.fills)
            fill_err += [abs(f - real[t]) for t, f in fills if t in real]
    starts_l = {lv: sp.load_start(cfg, lv) for lv in {o[0] for o in OUTCOMES}}
    outs = [outcome(cfg, lv, run, att, world, starts_l) for lv, run, att, _ in OUTCOMES]
    n = sum(conf.values())
    mean = (lambda v: round(sum(v) / len(v), 1) if v else None)
    return {
        "world": world, "episodes": n, "skipped": skipped,
        "left_both": conf[(True, True)], "left_real_only": conf[(True, False)],
        "left_sim_only": conf[(False, True)], "left_neither": conf[(False, False)],
        "agreement": round((conf[(True, True)] + conf[(False, False)]) / n, 3) if n else None,
        "lifetime_bias_s": mean(life_err),
        "lifetime_mae_s": mean([abs(x) for x in life_err]),
        "lifetime_bias_by_share": {k: mean(v) for k, v in sorted(per_share.items())},
        "fill_mae": mean(fill_err),
        "outcomes": outs,
        "outcomes_right": sum(o["sim"] == ("win" if o["real"] == "win" else "fail")
                              for o in outs),
    }


def report(r: Dict[str, Any]) -> List[str]:
    lines = [f"world {r['world'] or 'model'}: {r['episodes']} bucket episodes "
             f"({r['skipped']} skipped)",
             f"  left within the horizon: both {r['left_both']}, real only "
             f"{r['left_real_only']} (sim too slow / stuck), sim only {r['left_sim_only']}, "
             f"neither {r['left_neither']}; agreement {r['agreement']}",
             f"  departure time sim - real: bias {r['lifetime_bias_s']} s, MAE "
             f"{r['lifetime_mae_s']} s, by contact share at the tap "
             f"{r['lifetime_bias_by_share']}; fill MAE {r['fill_mae']} pts",
             f"  outcomes right {r['outcomes_right']}/{len(r['outcomes'])}:"]
    for o in r["outcomes"]:
        lines.append(f"    {o['level']} {o['run']} #{o['attempt']}: real {o['real']}, sim "
                     f"{o['sim']} ({o['blocked']}/{o['taps']} taps blocked, collected "
                     f"{o['collected']:.0%}; belt last collected at {o['sim_end_s']} s, real "
                     f"level {o['real_end_s']} s)")
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    from config import load_config

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--world", action="append", default=[], metavar="FIELD=VALUE")
    ap.add_argument("--horizon", type=float, default=40.0)
    ap.add_argument("--out", help="JSON (default debug/realcal/<stamp>.json)")
    args = ap.parse_args(argv)
    cfg = load_config()
    root = Path(cfg.paths.debug)
    runs = ([Path(r) for r in args.runs] if args.runs else
            sorted(p for p in (root / "runs").iterdir()
                   if p.is_dir() and p.name >= "20261021-000105"))   # 2x belt runs
    world = {k: float(v) for k, v in (s.split("=", 1) for s in args.world)}
    t0 = time.monotonic()
    r = evaluate(cfg, runs, world, args.horizon)
    for line in report(r):
        print(line)
    out = Path(args.out) if args.out else root / "realcal" / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=1), encoding="utf-8")
    print(f"-> {out} ({time.monotonic() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
