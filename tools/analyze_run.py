"""Post-mortem of a bot run (debug/runs/<stamp>/, written by bot.py / runlog.py).

    python -m tools.analyze_run                 # the latest run
    python -m tools.analyze_run debug/runs/20261017-101500

Prints a short report to paste back: result, taps, when the belt filled up for good, buckets
that stalled on the belt (and the tap that released them, with the exposure it had then), the
tap that took the last free slot, the belt at the end and the colours it was waiting for.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from runlog import load_events


def latest_run(root: Path) -> Optional[Path]:
    runs = sorted((p for p in (root / "runs").glob("*") if (p / "events.jsonl").exists()),
                  key=lambda p: p.name)
    return runs[-1] if runs else None


def _pct(x: Optional[float]) -> str:
    return "?" if x is None else f"{x:.0%}"


def _belt(rec: Dict[str, Any]) -> str:
    conv = rec.get("conveyor") or []
    return ", ".join(f"{b['color']}{'/' + '/'.join(b['stages']) if b.get('stages') else ''}"
                     f" {'?' if b['fill'] is None else str(b['fill']) + '%'}" for b in conv) or "-"


def _tap_line(tap: Dict[str, Any]) -> str:
    r, c = tap["pos"]
    return (f"#{tap['n']} t={tap['t']:.1f}s r{r}c{c} {tap['color']}"
            f"{'/' + '/'.join(tap['stages']) if tap.get('stages') else ''}: "
            f"k {tap['k_before']}->{'' if tap['k_before'] is None else tap['k_before'] + 1}, "
            f"avail {_pct(tap['avail'])} (exp {_pct(tap['exposed'])}), score {tap['score']:.2f}, "
            f"{tap['alternatives']} other allowed, {tap['why']}")


def stalls(analyses: List[Dict[str, Any]], stall_s: float) -> Dict[str, Tuple[float, float]]:
    """colour -> (longest stall in s, t when it began) from the belt stall clocks."""
    out: Dict[str, Tuple[float, float]] = {}
    for a in analyses:
        for color, s in (a.get("stalled") or {}).items():
            if s >= stall_s and s > out.get(color, (0.0, 0.0))[0]:
                out[color] = (s, a["t"] - s)
    return out


def full_since(analyses: List[Dict[str, Any]], slots: int) -> Optional[float]:
    """Time from which the counter read k = slots until the end (None if not full at the end)."""
    since = None
    for a in analyses:
        k = a.get("k")
        if k is None:
            continue
        if k >= slots:
            since = a["t"] if since is None else since
        else:
            since = None
    return since


def post_mortem(run_dir: Path) -> List[str]:
    events = load_events(run_dir)
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    run = json.loads((run_dir / "run.json").read_text()) if (run_dir / "run.json").exists() else {}
    cfg = run.get("config") or {}
    slots = (cfg.get("strategy") or {}).get("conveyor_slots", 5)
    stall_s = (cfg.get("strategy") or {}).get("stall_s", 4.0)
    analyses = [e for e in events if e["type"] == "analysis" and e.get("k") is not None]
    taps = [e for e in events if e["type"] == "tap"]
    effects = [e for e in events if e["type"] == "effect"]
    out = [f"run {run_dir.name} (commit {run.get('commit')}, "
           f"{'dry-run' if (run.get('args') or {}).get('dry_run') else 'live'})"]
    if result is None:
        out.append("result: none (run crashed or still running)")
    else:
        out.append(f"result: {result['result'].upper()} - {result['reason']} after "
                   f"{result['duration_s']:.0f}s, {result.get('taps', len(taps))} taps")
    fails = sum(1 for e in events if e["type"] == "analysis" and e.get("failure"))
    reads = sum(1 for e in events if e["type"] == "analysis")
    no_eff = sum(1 for e in effects if not e["ok"])
    out.append(f"analyses {reads}, read failures {fails}, taps without effect {no_eff}")
    ends = [e for e in events if e["type"] == "level_end"]
    if ends:
        out.append("levels: " + ", ".join(f"L{e['level']} {e['result']}" for e in ends))
    screen_taps = [e for e in events if e["type"] == "screen_tap"]
    if screen_taps:
        out.append(f"screen taps ({len(screen_taps)}):")
        out += [f"  {e['t']:7.1f}s {e['screen']}: {e['button']} at {e['rel']}"
                f"{'' if e['executed'] else ' [dry-run]'}  {e['frame_png']}"
                for e in screen_taps[-10:]]
    if not analyses:
        return out + ["no gameplay analysed"]

    out.append("taps:")
    out += ["  " + _tap_line(t) for t in taps[-8:]]
    if len(taps) > 8:
        out.insert(-8, f"  ... {len(taps) - 8} earlier taps")

    st = stalls(analyses, stall_s)
    if st:
        out.append(f"stalled on the belt (fill not growing >= {stall_s:.0f}s):")
        for color, (s, began) in sorted(st.items(), key=lambda kv: -kv[1][0]):
            rel = [t for t in taps if t["color"] == color and t["t"] <= began]
            src = (f"released by tap #{rel[-1]['n']} at {rel[-1]['t']:.1f}s with avail "
                   f"{_pct(rel[-1]['avail'])}" if rel else "not released by this run")
            at = min(analyses, key=lambda a: abs(a["t"] - began))
            out.append(f"  {color}: {s:.0f}s from {began:.1f}s, {src}; at the stall "
                       f"exposed {_pct(at['exposed'].get(color, 0.0))}, "
                       f"sand left {_pct(at['sand'].get(color, 0.0))}")

    last_slot = [t for t in taps if t.get("free_before") == 1]
    if last_slot:
        t = last_slot[-1]
        out.append("last free slot taken by: " + _tap_line(t))

    since = full_since(analyses, slots)
    end = analyses[-1]
    if since is not None:
        before = [t for t in taps if t["t"] <= since]
        out.append(f"belt full ({slots}/{slots}) from {since:.1f}s to the end"
                   + (f", last tap before: #{before[-1]['n']} {before[-1]['color']}"
                      if before else ""))
    out.append(f"belt at the end: {_belt(end)}")
    waiting = [b for b in end.get("conveyor") or [] if b["fill"] is None or b["fill"] < 100]
    for b in waiting:
        c = b["color"]
        out.append(f"  {c}: exposed {_pct(end['exposed'].get(c, 0.0))}, sand left "
                   f"{_pct(end['sand'].get(c, 0.0))}"
                   + (f", stalled {end['stalled'][c]:.0f}s" if c in (end.get("stalled") or {})
                      else ""))
    top = sorted(end["exposed"].items(), key=lambda kv: -kv[1])[:4]
    out.append("contact layer at the end: " + ", ".join(f"{c} {v:.0%}" for c, v in top))
    out.append(f"last decision: {end.get('reason')}")
    if end.get("grid"):
        out += ["grid at the end:"] + ["  " + row for row in end["grid"]]
    out += verdict(result, st, last_slot, since, waiting, end)
    return out


def verdict(result: Optional[Dict[str, Any]], st: Dict[str, Tuple[float, float]],
            last_slot: List[Dict[str, Any]], since: Optional[float],
            waiting: List[Dict[str, Any]], end: Dict[str, Any]) -> List[str]:
    if result is None or result["result"] not in ("fail", "revive"):
        return []
    out = ["verdict:"]
    buried = [b["color"] for b in waiting if end["exposed"].get(b["color"], 0.0) < 0.03]
    if buried:
        out.append(f"  belt buckets waiting for buried sand: {', '.join(buried)}")
    if last_slot:
        t = last_slot[-1]
        out.append(f"  tap #{t['n']} ({t['color']}, avail {_pct(t['avail'])}) took the last "
                   "slot" + (" - its colour then stalled" if t["color"] in st else ""))
    if st:
        worst = max(st.items(), key=lambda kv: kv[1][0])
        out.append(f"  longest stall: {worst[0]} {worst[1][0]:.0f}s")
    if since is not None:
        out.append(f"  no free slot from {since:.1f}s on")
    if len(out) == 1:
        out.append("  no stall, last-slot tap or buried belt colour found: check the frames")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.analyze_run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", default=None, help="run directory (default: latest)")
    ap.add_argument("--debug", default="debug", help="debug root (for the latest run)")
    args = ap.parse_args(argv)
    run_dir = Path(args.run) if args.run else latest_run(Path(args.debug))
    if run_dir is None or not (run_dir / "events.jsonl").exists():
        print("no run found", file=sys.stderr)
        return 1
    print("\n".join(post_mortem(run_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
