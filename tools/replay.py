"""Replay harness (redesign phase A): how well does the simulator predict the real game?

From every saved gameplay frame of every recorded run (debug/runs/*/frames, re-read with
today's vision) the simulator (sim.Sim, the level planner's model) runs forward with the taps
the bot really made, at their real times, and is compared with what the bot really read in the
frames that followed (events.jsonl "analysis", ~every 1.2 s):

* **belt fill** - per colour the real read fills (< 100 %) and the simulated ones, both sorted,
  paired; absolute error in % points, by time since the start frame;
* **k** - buckets on the belt, real counter vs simulated;
* **sand** - per colour the share of its sand still in the picture (relative to the start
  frame: real vision shares vs simulated cells), absolute error;
* **exposed** - per colour the contact-layer share, vision vs simulated (at 0 s it measures the
  picture model alone);
* **gains in 10 s** - the decision that matters for waiting: does a bucket on the belt at the
  start frame gain sand within ``gain_s``? Real = a fill read later is >= ``gain_min`` points
  above the fill it grew from (buckets only grow, so a reading is matched to the highest start
  fill at or below it - two same-colour buckets read alternately are no gain); simulated = its
  slot collected anything. "Optimistic" = simulated gain while the real bucket stalled.

A trajectory stops at the horizon, at a tap the simulator cannot follow (its cell is not a
bucket of known colour there: misread, a tunnel drop, an opened lock) and at the end of
gameplay (level end, popup, other screen).

    python -m tools.replay                          # all runs, report + debug/replay/<stamp>.json
    python -m tools.replay debug/runs/20261021-*    # some runs
    python -m tools.replay --set fill_rate=80 --set contact_layer=0.03   # other sim parameters

``--set`` takes planner fields (planner: in config.yaml) and vision fields (those need
``--rebuild``: the cached frame reads hold the pictures). Frame reads are cached in
debug/replay/cache (``--rebuild`` after vision changes).
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

CACHE_VERSION = 1
BINS = ((0, 1), (1, 5), (5, 10), (10, 20), (20, 31))


@dataclass
class Start:
    run: str
    frame: int
    t: float
    path: Path
    x2: bool


@dataclass
class Traj:
    """Comparisons of one trajectory (one start frame)."""
    start: Start
    end: str = "horizon"                                   # why it stopped
    fills: List[Tuple[float, float]] = field(default_factory=list)       # (dt, |error| pts)
    k: List[Tuple[float, int]] = field(default_factory=list)             # (dt, sim - real)
    sand: List[Tuple[float, float]] = field(default_factory=list)        # (dt, |error|)
    exposed: List[Tuple[float, float]] = field(default_factory=list)     # (dt, |error|)
    gains: List[Tuple[str, bool, bool]] = field(default_factory=list)    # (colour, sim, real)
    taps: int = 0


# --- pure comparisons (tests/test_replay.py) ---------------------------------------------------


def pair_fills(real: Sequence[Tuple[str, Optional[int]]], sim: Sequence[Tuple[str, int]]
               ) -> List[float]:
    """|sim - real| fill per bucket: per colour the read fills below 100 and the simulated
    ones, both sorted high first, paired in order (unpaired buckets count in ``k``)."""
    by_real: Dict[str, List[int]] = defaultdict(list)
    by_sim: Dict[str, List[int]] = defaultdict(list)
    for c, f in real:
        if c and f is not None and f < 100:
            by_real[c].append(f)
    for c, f in sim:
        by_sim[c].append(max(0, min(100, f)))
    out = []
    for c, fs in by_real.items():
        for a, b in zip(sorted(fs, reverse=True), sorted(by_sim.get(c, []), reverse=True)):
            out.append(float(abs(a - b)))
    return out


def real_gain(start: Sequence[int], later: Sequence[int], min_gain: int = 3) -> bool:
    """Did a bucket of the colour gain sand? ``start``: its buckets' fills at the start frame,
    ``later``: every fill of the colour read since. Each reading grew from the highest start
    fill at or below it (+1 for label jitter)."""
    if not start:
        return False
    for f in later:
        origins = [s for s in start if s <= f + 1]
        if origins and f - max(origins) >= min_gain:
            return True
    return False


def summarize(trajs: Sequence[Traj]) -> Dict[str, Any]:
    def binned(rows):
        out = {}
        for lo, hi in BINS:
            vals = [abs(v) for dt, v in rows if lo <= dt < hi]
            out[f"{lo}-{hi}s"] = {"n": len(vals),
                                  "mae": round(sum(vals) / len(vals), 3) if vals else None}
        return out

    allk = [(dt, v) for t in trajs for dt, v in t.k]
    k_exact = {}
    for lo, hi in BINS:
        vals = [v for dt, v in allk if lo <= dt < hi]
        k_exact[f"{lo}-{hi}s"] = round(sum(v == 0 for v in vals) / len(vals), 3) if vals else None
    k_bias = {}                     # mean sim - real: > 0 = the sim has buckets the belt lost
    for lo, hi in BINS:
        vals = [v for dt, v in allk if lo <= dt < hi]
        k_bias[f"{lo}-{hi}s"] = round(sum(vals) / len(vals), 3) if vals else None
    g = [x for t in trajs for x in t.gains]
    tp = sum(s and r for _, s, r in g)
    fp = sum(s and not r for _, s, r in g)
    fn = sum(not s and r for _, s, r in g)
    tn = sum(not s and not r for _, s, r in g)
    ends: Dict[str, int] = defaultdict(int)
    for t in trajs:
        ends[t.end] += 1
    return {
        "trajectories": len(trajs),
        "ends": dict(ends),
        "fill_mae": binned([x for t in trajs for x in t.fills]),
        "k_mae": binned(allk),
        "k_exact": k_exact,
        "k_bias": k_bias,
        "sand_mae": binned([x for t in trajs for x in t.sand]),
        "exposed_mae": binned([x for t in trajs for x in t.exposed]),
        "gain": {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                 "accuracy": round((tp + tn) / len(g), 3) if g else None,
                 "gain_recall": round(tp / (tp + fn), 3) if tp + fn else None,
                 "gain_precision": round(tp / (tp + fp), 3) if tp + fp else None,
                 "stall_recall": round(tn / (tn + fp), 3) if tn + fp else None},
    }


# --- runs ----------------------------------------------------------------------------------------


def starts_of(run_dir: Path, events: List[Dict[str, Any]]) -> List[Start]:
    """Saved raw frames (not the annotated copies) whose analysis read gameplay cleanly."""
    import csv

    frames = {}
    for p in (run_dir / "frames").glob("*.png"):
        if "annotated" not in p.name and p.name[:5].isdigit():
            frames.setdefault(int(p.name[:5]), p)
    speed: Dict[int, bool] = {}
    belt = run_dir / "belt.csv"
    if belt.exists():
        for row in csv.DictReader(belt.open(encoding="utf-8")):
            if row.get("frame") and row.get("speed"):
                speed[int(row["frame"])] = row["speed"] == "2"
    out = []
    for e in events:
        if (e["type"] == "analysis" and e.get("screen") == "gameplay" and not e.get("error")
                and e.get("frame") in frames and e.get("k") is not None):
            out.append(Start(run_dir.name, e["frame"], e["t"], frames[e["frame"]],
                             speed.get(e["frame"], False)))
    seen, uniq = set(), []
    for s in out:
        if s.frame not in seen:
            seen.add(s.frame)
            uniq.append(s)
    return uniq


class Reader:
    """Frame -> (LevelState, sim.Picture), cached on disk."""

    def __init__(self, cfg, cache: Path, rebuild: bool = False) -> None:
        self.cfg, self.cache, self.rebuild = cfg, cache, rebuild
        self._an = None

    def read(self, start: Start):
        f = self.cache / start.run / f"{start.frame:05d}.pkl"
        if f.exists() and not self.rebuild:
            # pickle is safe here: the cache holds only files this tool wrote itself
            with f.open("rb") as fh:
                ver, state, pic = pickle.load(fh)
            if ver == CACHE_VERSION:
                return state, pic
        import cv2

        from analyzer import Analyzer
        from levelplan import _palette
        from sim import picture_from_frame

        if self._an is None:
            self._an = Analyzer(self.cfg)
        img = cv2.imread(str(start.path))
        res = self._an.analyze(img, log=lambda s: None, now=0.0) if img is not None else None
        state = pic = None
        if res is not None and res.state is not None:
            state = res.state
            pic = picture_from_frame(img, _palette(self.cfg), self.cfg.vision,
                                     self.cfg.planner.cell_px, state.shells, state.pic_locks)
        f.parent.mkdir(parents=True, exist_ok=True)
        with f.open("wb") as fh:
            pickle.dump((CACHE_VERSION, state, pic), fh)
        return state, pic


def replay(start: Start, events: List[Dict[str, Any]], state, pic, pc, horizon: float,
           gain_s: float = 10.0, gain_min: int = 3) -> Traj:
    """Run the simulator from ``start`` with the real taps; compare with the real reads."""
    from cells import Kind
    from levelplan import sim_config
    from sim import Sim

    tr = Traj(start)
    sim = Sim.start(state, pic, sim_config(pc, state.conveyor_slots, start.x2))
    sim.grid.disagreements = []
    sim.grid.tappable = sim.grid.playable()
    base = next(e for e in events if e["type"] == "analysis" and e.get("frame") == start.frame)
    slots0 = {id(s): (s, s.stage, s.left) for s in sim.belt}
    names = [n for n in pic.names]
    cnt0 = {c: pic.count(c) for c in names}
    start_fills: Dict[str, List[int]] = defaultdict(list)
    for b in base.get("conveyor") or []:
        if b.get("color") and b.get("fill") is not None and b["fill"] < 100:
            start_fills[b["color"]].append(b["fill"])
    later: Dict[str, List[int]] = defaultdict(list)
    sim_gain: Dict[str, bool] = {}
    covered = 0.0

    def compare(e: Dict[str, Any], dt: float) -> None:
        real = [(b.get("color"), b.get("fill")) for b in e.get("conveyor") or []]
        for err in pair_fills(real, sim.belt_fills()):
            tr.fills.append((dt, err))
        tr.k.append((dt, len(sim.belt) - e["k"]))
        for c, s0 in (base.get("sand") or {}).items():
            if s0 >= 0.01 and cnt0.get(c):
                tr.sand.append((dt, abs((e.get("sand") or {}).get(c, 0.0) / s0
                                        - sim.pic.count(c) / cnt0[c])))
        exp = e.get("exposed") or {}
        for c in set(exp) | {c for c in names if sim.pic.share(c) > 0}:
            tr.exposed.append((dt, abs(exp.get(c, 0.0) - sim.pic.share(c))))

    compare(base, 0.0)
    for e in events:
        if e["t"] <= start.t:
            continue
        dt = e["t"] - start.t
        if dt > horizon:
            break
        if e["type"] in ("level_start", "level_end", "stop"):
            tr.end = e["type"]
            break
        if e["type"] not in ("tap", "analysis"):
            continue
        while sim.t < dt:
            sim.lap()
        if dt <= gain_s:
            for s, stage, left in slots0.values():
                c = s.stages[stage]
                moved = s not in sim.belt or s.stage != stage or s.left < left
                sim_gain[c] = sim_gain.get(c, False) or moved
        if e["type"] == "tap":
            if not e.get("executed", True):
                continue
            pos = tuple(e["pos"])
            g = sim.grid.grid
            inside = 0 <= pos[0] < len(g) and 0 <= pos[1] < len(g[pos[0]])
            cell = sim.grid.cell(pos) if inside else None   # another lattice read today
            if cell is None or cell.kind is not Kind.BUCKET or not cell.color:
                tr.end = "unknown tap"
                break
            sim.place(pos)
            tr.taps += 1
            continue
        if e.get("screen") != "gameplay":
            tr.end = f"screen {e.get('screen')}"
            break
        if e.get("error") or e.get("k") is None:
            continue
        compare(e, dt)
        if dt <= gain_s:
            covered = dt
            for b in e.get("conveyor") or []:
                if b.get("color") and b.get("fill") is not None:
                    later[b["color"]].append(b["fill"])
    if covered >= gain_s - 1.5:
        for c, fills in start_fills.items():
            if c in sim_gain:
                tr.gains.append((c, sim_gain[c], real_gain(fills, later[c], gain_min)))
    return tr


def _apply(cfg, sets: Sequence[str]) -> None:
    for s in sets:
        name, value = s.split("=", 1)
        for sect in (cfg.planner, cfg.vision):
            if hasattr(sect, name):
                old = getattr(sect, name)
                setattr(sect, name, type(old)(value) if old is not None else float(value))
                break
        else:
            raise SystemExit(f"--set {name}: no such planner / vision field")


def run_all(cfg, run_dirs: Sequence[Path], horizon: float, rebuild: bool = False,
            log=print) -> Tuple[List[Traj], Dict[str, Any]]:
    from runlog import load_events

    reader = Reader(cfg, Path(cfg.paths.debug) / "replay" / "cache", rebuild)
    trajs: List[Traj] = []
    per_run: Dict[str, Any] = {}
    for rd in run_dirs:
        if not (rd / "events.jsonl").exists() or not (rd / "frames").is_dir():
            continue
        events = load_events(rd)
        mine = []
        for st in starts_of(rd, events):
            state, pic = reader.read(st)
            if state is None or pic is None:
                continue
            mine.append(replay(st, events, state, pic, cfg.planner, horizon))
        if mine:
            trajs += mine
            per_run[rd.name] = summarize(mine)
            log(f"{rd.name}: {len(mine)} trajectories")
    return trajs, per_run


def _fmt(x) -> str:
    return "-" if x is None else f"{x:.3g}"


def report(total: Dict[str, Any], per_run: Dict[str, Any]) -> List[str]:
    g = total["gain"]
    lines = [f"trajectories {total['trajectories']} (ends: {total['ends']})",
             "             " + " ".join(f"{b:>9s}" for b in total["fill_mae"])]
    for key, label in (("fill_mae", "fill MAE pts"), ("k_mae", "k MAE"),
                       ("k_exact", "k exact"), ("k_bias", "k bias"), ("sand_mae", "sand MAE"),
                       ("exposed_mae", "exposed MAE")):
        row = total[key]
        vals = [row[b] if key in ("k_exact", "k_bias") else row[b]["mae"] for b in row]
        lines.append(f"{label:13s}" + " ".join(f"{_fmt(v):>9s}" for v in vals))
    lines.append(f"gains in 10 s: accuracy {_fmt(g['accuracy'])}, gain recall "
                 f"{_fmt(g['gain_recall'])}, gain precision {_fmt(g['gain_precision'])}, "
                 f"stall recall {_fmt(g['stall_recall'])} (tp {g['tp']} fp {g['fp']} "
                 f"fn {g['fn']} tn {g['tn']}; fp = sim optimistic)")
    lines.append("per run: fill MAE 5-10 s | 10-20 s, sand MAE 10-20 s, gain accuracy, "
                 "stall recall")
    for run, s in per_run.items():
        lines.append(f"  {run}: {_fmt(s['fill_mae']['5-10s']['mae'])} | "
                     f"{_fmt(s['fill_mae']['10-20s']['mae'])}, "
                     f"{_fmt(s['sand_mae']['10-20s']['mae'])}, {_fmt(s['gain']['accuracy'])}, "
                     f"{_fmt(s['gain']['stall_recall'])} ({s['trajectories']} traj)")
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    from config import load_config

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="*", help="run dirs (default: all of debug/runs)")
    ap.add_argument("--horizon", type=float, default=30.0, help="seconds per trajectory")
    ap.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                    help="override a planner / vision field")
    ap.add_argument("--rebuild", action="store_true", help="re-read the frames (vision changed)")
    ap.add_argument("--out", help="report JSON (default debug/replay/<stamp>.json)")
    args = ap.parse_args(argv)
    cfg = load_config()
    _apply(cfg, args.set)
    runs = ([Path(r) for r in args.runs] if args.runs
            else sorted(p for p in (Path(cfg.paths.debug) / "runs").iterdir() if p.is_dir()))
    t0 = time.monotonic()
    trajs, per_run = run_all(cfg, runs, args.horizon, args.rebuild)
    total = summarize(trajs)
    for line in report(total, per_run):
        print(line)
    out = Path(args.out) if args.out else (
        Path(cfg.paths.debug) / "replay" / f"{time.strftime('%Y%m%d-%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"set": args.set, "horizon": args.horizon, "total": total,
                               "runs": per_run}, indent=1), encoding="utf-8")
    print(f"-> {out} ({time.monotonic() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
