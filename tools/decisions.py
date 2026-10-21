"""Old vs new decisions on the recorded boards (redesign phase C report).

For every saved gameplay frame of a run (frames/ and the 1 s crops, as tools/replay.py) the
search decider (decider.py) decides on that frame with the belief it would have had: every
earlier analysis of the level fed to it (belt readings, k counter, times) and every real tap.
Its decision is put next to what the old decider did on that frame (events.jsonl "analysis":
reason / move). The new decider's own choices are not played (the run is what it is), so
from the first disagreement on it decides on boards the old one led to.

    python -m tools.decisions                       # the L187 / L189 / L190 runs
    python -m tools.decisions debug/runs/20261021-143729 --every 3

Summary per run: same tap, other tap, old waits / new taps (and whether a belt bucket was
gaining - the new decider may wait only then), old taps / new waits; the frames listed in
debug/decisions/<run>.txt.
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

RUNS = {"L187": ["20261020-224631", "20261021-000105"],
        "L189": ["20261021-092045", "20261021-101655", "20261021-112241"],
        "L190": ["20261021-122825", "20261021-143729"]}


def _buckets(rec: Dict[str, Any]):
    from state import ConveyorBucket

    if rec.get("k") == 0:
        return []
    return [ConveyorBucket(b.get("color"), b.get("fill"), False, tuple(b.get("stages") or ()))
            for b in rec.get("conveyor") or []]


def run_decisions(cfg, run_dir: Path, every: int = 1, log=print) -> List[Dict[str, Any]]:
    from decider import SearchDecider
    from runlog import load_events
    from tools.replay import Reader, starts_of

    events = load_events(run_dir)
    starts = {s.frame: s for s in starts_of(run_dir, events)}
    reader = Reader(cfg, Path(cfg.paths.debug) / "replay" / "cache", cell_px=cfg.planner.cell_px)
    dec: Optional[SearchDecider] = None
    out: List[Dict[str, Any]] = []
    n = 0
    for e in events:
        if e["type"] == "level_start" or dec is None:
            dec = SearchDecider(cfg)
            dec.signature = f"{run_dir.name}-{e.get('level', 0)}"   # no memory: no leaks
            if e["type"] == "level_start":
                continue
        if e["type"] == "tap" and e.get("executed", True):
            dec.tapped(tuple(e["pos"]), e["color"], tuple(e.get("stages") or ()), e["t"])
            continue
        if (e["type"] != "analysis" or e.get("screen") != "gameplay" or e.get("error")
                or e.get("k") is None):
            continue
        dec.belief.update(_buckets(e), e["k"], e["t"])
        st = starts.get(e.get("frame"))
        if st is None:
            continue
        n += 1
        if (n - 1) % every and e.get("move") is None:
            continue                  # every tap frame, every n-th wait frame
        state, pic = reader.read(st)
        if state is None or pic is None:
            continue
        if e.get("tappable") is not None:   # the frame's perception guards, as recorded
            state.tappable = [tuple(p) for p in e["tappable"]
                              if tuple(p) in set(state.tappable) | set(state.playable())]
        dec.x2 = st.x2
        d = dec.decide(state, lambda: pic.copy(), e["t"], log=lambda s: None)
        old = e.get("move")
        out.append({
            "t": round(e["t"], 1), "frame": e["frame"], "k": e["k"],
            "gaining": dec.belief.gaining(e["t"]), "belt": dec.belief.describe(e["t"]),
            "old": (tuple(old["pos"]), old["color"]) if old else None,
            "old_reason": e.get("reason") or "",
            "new": (d.move.pos, d.move.color) if d.move else None, "new_reason": d.reason})
    return out


def classify(r: Dict[str, Any]) -> str:
    old, new = r["old"], r["new"]
    if old is None and new is None:
        return "both wait"
    if old is not None and new is not None:
        return "same tap" if tuple(old[0]) == tuple(new[0]) else "other tap"
    if old is None:
        return "old waits, new taps" + ("" if r["gaining"] else " (belt not gaining)")
    return "old taps, new waits"


def report(name: str, rows: Sequence[Dict[str, Any]]) -> List[str]:
    kinds = Counter(classify(r) for r in rows)
    lines = [f"{name}: {len(rows)} frames - " + ", ".join(f"{k} {v}" for k, v in
                                                       kinds.most_common())]
    idle = [r for r in rows if r["old"] is None and not r["gaining"] and r["new"] is not None]
    if idle:
        lines.append(f"  old waited on {len(idle)} frames while no belt bucket was gaining "
                     f"(t {idle[0]['t']:.0f}-{idle[-1]['t']:.0f}s); e.g. "
                     f"\"{idle[0]['old_reason'][:90]}\"")
    return lines


def _job(args):
    from config import load_config

    path, every = args
    cfg = load_config()
    return path.name, run_decisions(cfg, path, every)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from config import load_config

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--every", type=int, default=2, help="every n-th wait frame (taps: all)")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args(argv)
    cfg = load_config()
    root = Path(cfg.paths.debug)
    level_of = {r: lv for lv, rs in RUNS.items() for r in rs}
    runs = [Path(r) for r in args.runs] or [root / "runs" / r for rs in RUNS.values() for r in rs]
    t0 = time.monotonic()
    jobs = [(p, args.every) for p in runs if (p / "events.jsonl").exists()]
    if args.jobs > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(args.jobs) as pool:
            results = pool.map(_job, jobs, chunksize=1)
    else:
        results = [_job(j) for j in jobs]
    out_dir = root / "decisions"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in results:
        for line in report(f"{level_of.get(name, '?')} {name}", rows):
            print(line)
        with (out_dir / f"{name}.txt").open("w", encoding="utf-8") as fh:
            for r in rows:
                o = f"r{r['old'][0][0]}c{r['old'][0][1]} {r['old'][1]}" if r["old"] else "wait"
                n = f"r{r['new'][0][0]}c{r['new'][0][1]} {r['new'][1]}" if r["new"] else "wait"
                fh.write(f"{r['t']:7.1f}s #{r['frame']:05d} k{r['k']} [{classify(r)}] old {o} | "
                         f"new {n}\n    belt: {r['belt']}\n    old: {r['old_reason'][:160]}\n"
                         f"    new: {r['new_reason'][:200]}\n")
    print(f"-> {out_dir} ({time.monotonic() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
