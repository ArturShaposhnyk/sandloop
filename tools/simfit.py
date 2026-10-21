"""Phase D: fit the simulator to the real game (tools/realcal.py's checks).

Grid search over the 2x belt's SimConfig fields and the contact depth. Ranked by: real
outcomes reproduced (won attempts win, lost ones lose), then per-bucket agreement ("left the
belt within the horizon"), then the departure-time error.

    python -m tools.simfit --jobs 8
    python -m tools.simfit --grid reach_delay_s=1.5,2.5 --grid contact=2,2.5
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

GRID: Dict[str, List[float]] = {
    "reach_delay_s": [1.5, 2.5, 3.5],
    "fill_rate": [150.0, 300.0, 600.0],
    "fill_max": [60.0, 100.0],
    "fill_min_exposed": [0.0, 0.02],
    "contact": [1.5, 2.0, 2.5, 3.0],
    "finish_tol": [0.03],
}


def _job(world: Dict[str, float]) -> Dict[str, Any]:
    from config import load_config
    from tools.realcal import evaluate

    cfg = load_config()
    runs = sorted(p for p in (Path(cfg.paths.debug) / "runs").iterdir()
                  if p.is_dir() and p.name >= "20261021-000105")
    r = evaluate(cfg, runs, world, log=lambda s: None)
    return r


def key(r: Dict[str, Any]):
    return (r["outcomes_right"], r["agreement"] or 0.0, -(r["lifetime_mae_s"] or 99.0))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--grid", action="append", default=[], metavar="FIELD=V1,V2,...")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", help="JSON (default debug/realcal/fit-<stamp>.json)")
    args = ap.parse_args(argv)
    grid = dict(GRID)
    for g in args.grid:
        name, vals = g.split("=", 1)
        grid[name] = [float(v) for v in vals.split(",")]
    worlds = [dict(zip(grid, vals)) for vals in itertools.product(*grid.values())]
    t0 = time.monotonic()
    import multiprocessing as mp

    with mp.get_context("spawn").Pool(args.jobs) as pool:
        results = pool.map(_job, worlds, chunksize=1)
    results.sort(key=key, reverse=True)
    for r in results[:args.top]:
        wins = [o for o in r["outcomes"] if o["real"] == "win"]
        print(f"{r['outcomes_right']}/{len(r['outcomes'])} outcomes, agreement {r['agreement']}, "
              f"departure MAE {r['lifetime_mae_s']} s (bias {r['lifetime_bias_s']}), real wins "
              + ", ".join(f"{o['level']} {o['sim']} ({o['blocked']} blocked, "
                          f"{o['collected']:.0%})" for o in wins) + f"  {r['world']}")
    from config import load_config

    out = Path(args.out) if args.out else (
        Path(load_config().paths.debug) / "realcal" / f"fit-{time.strftime('%Y%m%d-%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"-> {out} ({len(worlds)} worlds, {time.monotonic() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
