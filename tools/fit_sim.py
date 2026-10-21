"""Fit the simulator's belt / sand parameters to the recorded runs (redesign phase B).

Loads every replay trajectory once (tools/replay.py: start frame, its events, the frame's
state and picture from the cache), then evaluates parameter sets in parallel and searches
them coordinate by coordinate (a grid per parameter, a few rounds).

Objective (higher is better): the balanced accuracy of "gains sand within 10 s" (mean of gain
recall and stall recall, so predicting "never fills" scores 0.5, not 1) on all runs and on the
L189 runs alone (the level where the plans waited on stalled colours), minus
``fill_weight`` x the fill MAE at 5-20 s in % points.

    python -m tools.fit_sim                   # search, report before / after, best set -> JSON
    python -m tools.fit_sim --cv              # + 2-fold check: fit on half the runs, test on the rest
    python -m tools.fit_sim --eval fill_rate=90 contact=3     # one parameter set

Parameters: planner fields (fill_* for 1x, speed_x2_fill_* for 2x) and ``contact`` (picture
cells a bucket reaches per column; config: vision.contact_layer x frame height / cell_px).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tools.replay import Reader, replay, starts_of, summarize

# run -> level (docs/STATUS.md); earlier runs played L180-L186
LEVELS = {"20261021-001541": "L188", "20261021-092045": "L189", "20261021-101655": "L189",
          "20261021-112241": "L189", "20261021-122825": "L190", "20261021-143729": "L190"}


def level_of(run: str) -> str:
    if run in LEVELS:
        return LEVELS[run]
    return "L187" if run >= "20261020" else "L180-186"


GRID: Dict[str, List[float]] = {
    "fill_rate": [40, 60, 80, 100, 130, 160, 200],
    "fill_min_exposed": [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15],
    "fill_max": [15, 20, 30, 40, 60],
    "speed_x2_fill_rate": [40, 60, 90, 120, 160, 200, 260],
    "speed_x2_fill_min_exposed": [0.02, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.16],
    "speed_x2_fill_max": [20, 30, 45, 60, 80, 100],
    "contact": [1, 2, 3, 4, 5, 6, 8],
}

_ITEMS: List[Tuple[Any, ...]] = []        # (start, events, state, pic) - global for the workers
_PC: Any = None
HORIZON = 30.0


def load(cfg, runs: Sequence[Path], horizon: float = HORIZON, cell_px: Optional[int] = None
         ) -> List[Tuple[Any, ...]]:
    from runlog import load_events

    reader = Reader(cfg, Path(cfg.paths.debug) / "replay" / "cache", cell_px=cell_px)
    out = []
    for rd in runs:
        if not (rd / "events.jsonl").exists() or not (rd / "frames").is_dir():
            continue
        events = load_events(rd)
        for st in starts_of(rd, events):
            state, pic = reader.read(st)
            if state is None or pic is None:
                continue
            ev = [e for e in events if st.t - 0.01 <= e["t"] <= st.t + horizon + 1]
            out.append((st, ev, state, pic))
    return out


def _run(args: Tuple[int, Dict[str, float]]):
    i, params = args
    st, ev, state, pic = _ITEMS[i]
    fields = {k: v for k, v in params.items() if k != "contact"}
    pc = dataclasses.replace(_PC, **fields) if fields else _PC
    c = params.get("contact")
    return replay(st, ev, state, pic, pc, HORIZON, contact=None if c is None else int(c))


def evaluate(pool, idx: Sequence[int], params: Dict[str, float]) -> List[Any]:
    return pool.map(_run, [(i, params) for i in idx], chunksize=8)


def score(trajs, fill_weight: float = 0.005) -> Tuple[float, Dict[str, Any]]:
    total = summarize(trajs)
    l189 = summarize([t for t in trajs if level_of(t.start.run) == "L189"])

    def bal(s):
        g = s["gain"]
        if g["gain_recall"] is None or g["stall_recall"] is None:
            return 0.5
        return (g["gain_recall"] + g["stall_recall"]) / 2

    fills = [v for t in trajs for dt, v in t.fills if 5 <= dt < 20]
    mae = sum(fills) / len(fills) if fills else 0.0
    return (bal(total) + bal(l189)) / 2 - fill_weight * mae, {
        "total": total, "L189": l189, "fill_mae_5_20": round(mae, 2)}


def search(pool, idx, start: Dict[str, float], rounds: int = 3, log=print
           ) -> Tuple[Dict[str, float], float]:
    best = dict(start)
    best_s, _ = score(evaluate(pool, idx, best))
    for r in range(rounds):
        changed = False
        for name, values in GRID.items():
            for v in values:
                if best.get(name) == v:
                    continue
                trial = {**best, name: v}
                s, _ = score(evaluate(pool, idx, trial))
                if s > best_s + 1e-4:
                    best, best_s, changed = trial, s, True
            log(f"round {r + 1} {name}: {best.get(name)} (score {best_s:.4f})")
        if not changed:
            break
    return best, best_s


def table(name: str, info: Dict[str, Any]) -> List[str]:
    t, g = info["total"], info["total"]["gain"]
    lines = [f"{name}:  stall recall {g['stall_recall']}, gain recall {g['gain_recall']}, "
             f"accuracy {g['accuracy']} | L189 stall recall {info['L189']['gain']['stall_recall']},"
             f" gain recall {info['L189']['gain']['gain_recall']}"]
    lines.append("             " + " ".join(f"{b:>9s}" for b in t["fill_mae"]))
    for key, label in (("fill_mae", "fill MAE pts"), ("k_exact", "k exact"),
                       ("k_bias", "k bias"), ("sand_mae", "sand MAE"),
                       ("exposed_mae", "exposed MAE")):
        row = t[key]
        vals = [row[b] if key in ("k_exact", "k_bias") else row[b]["mae"] for b in row]
        lines.append(f"{label:13s}" + " ".join("-" if v is None else f"{v:>9.3g}" for v in vals))
    return lines


def defaults(pc, contact: int) -> Dict[str, float]:
    return {**{k: getattr(pc, k) for k in GRID if k != "contact"}, "contact": contact}


def main(argv: Optional[Sequence[str]] = None) -> int:
    global _ITEMS, _PC
    from config import load_config

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eval", nargs="*", metavar="FIELD=VALUE", help="evaluate one set")
    ap.add_argument("--cv", action="store_true", help="2-fold check by run")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--cell-px", type=int, default=None, help="picture block size (px)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = load_config()
    t0 = time.monotonic()
    runs = sorted(p for p in (Path(cfg.paths.debug) / "runs").iterdir() if p.is_dir())
    _ITEMS, _PC = load(cfg, runs, cell_px=args.cell_px), cfg.planner
    if args.cell_px:              # contact grid in the same pixels (4..64 px)
        k = cfg.planner.cell_px / args.cell_px
        GRID["contact"] = sorted({max(1, round(c * k)) for c in [0.5, 1, 2, 3, 4, 5, 6, 8]})
    contact = max(1, round(cfg.vision.contact_layer * 1272 / cfg.planner.cell_px))
    base = defaults(cfg.planner, _ITEMS[0][3].contact if _ITEMS else contact)
    print(f"{len(_ITEMS)} trajectories loaded ({time.monotonic() - t0:.0f}s)")
    idx = list(range(len(_ITEMS)))
    with mp.get_context("fork").Pool() as pool:
        if args.eval is not None:
            params = {**base, **{k: float(v) for k, v in (a.split("=", 1) for a in args.eval)}}
            s, info = score(evaluate(pool, idx, params))
            print("\n".join(table(f"score {s:.4f}", info)))
            return 0
        s0, info0 = score(evaluate(pool, idx, base))
        best, s1 = search(pool, idx, base, args.rounds)
        _, info1 = score(evaluate(pool, idx, best))
        print("\n".join(table(f"BEFORE {base} score {s0:.4f}", info0)))
        print("\n".join(table(f"AFTER {best} score {s1:.4f}", info1)))
        out: Dict[str, Any] = {"before": {"params": base, **info0},
                               "after": {"params": best, **info1}}
        if args.cv:
            runs_ = sorted({it[0].run for it in _ITEMS})
            folds = [set(runs_[0::2]), set(runs_[1::2])]
            out["cv"] = []
            for k, train in enumerate(folds):
                tr = [i for i in idx if _ITEMS[i][0].run in train]
                te = [i for i in idx if _ITEMS[i][0].run not in train]
                p, _ = search(pool, tr, base, args.rounds, log=lambda s: None)
                st, info = score(evaluate(pool, te, p))
                sb, info_b = score(evaluate(pool, te, base))
                print(f"cv fold {k + 1}: test runs {len(runs_) - len(train)}, "
                      f"before {sb:.4f} -> after {st:.4f}; stall recall "
                      f"{info_b['total']['gain']['stall_recall']} -> "
                      f"{info['total']['gain']['stall_recall']}, fill MAE 5-20 s "
                      f"{info_b['fill_mae_5_20']} -> {info['fill_mae_5_20']}")
                out["cv"].append({"params": p, "before": sb, "after": st})
    path = Path(args.out or Path(cfg.paths.debug) / "replay" / f"fit-{time.strftime('%Y%m%d-%H%M%S')}.json")
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"-> {path} ({time.monotonic() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
