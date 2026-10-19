"""Run log of bot.py: every run gets debug/runs/<stamp>/ with

    run.json         start time, args, git commit, config snapshot
    events.jsonl     one JSON object per line, "type": analysis | tap | effect | screen | log | stop
    belt.csv         belt over time: t, k, slots ("red 45|cyan/orange 10"), mean fill %
    frames/          PNGs: every tap (raw + annotated), decision changes, the final screen
    result.json      win | fail | revive | stopped, why, duration, taps
    bot.log          the console output

Read back by tools/analyze_run.py. JSON only for plain data (no numpy types).
"""
from __future__ import annotations

import csv
import dataclasses
import json
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    import numpy as np

    from analyzer import Analysis
    from strategy import Candidate


def _round(d: Dict[str, float], n: int = 4) -> Dict[str, float]:
    return {k: round(float(v), n) for k, v in d.items()}


def candidate_record(c: "Candidate") -> Dict[str, Any]:
    return {"pos": list(c.pos), "color": c.color, "score": round(c.score, 4),
            "rejected": c.rejected, "exposed": round(c.exposed, 4),
            "belt_demand": round(c.belt_demand, 4), "avail": round(c.avail, 4),
            "stages": list(c.stages), "short_stages": list(c.short_stages),
            "opens": [list(p) for p in c.opens], "on_belt": c.on_belt,
            "plan": c.plan.describe(1, detail=False) if c.plan is not None else None}


def analysis_record(res: "Analysis") -> Dict[str, Any]:
    rec: Dict[str, Any] = {"screen": res.screen, "failure": res.failure, "error": res.error}
    st, d = res.state, res.decision
    if st is not None:
        rec.update({
            "k": st.conveyor_k, "free": st.free_slots,
            "conveyor": [{"color": b.color, "fill": b.fill, "stages": list(b.stages)}
                         for b in st.conveyor],
            "sand": _round(st.sand), "exposed": _round(st.exposed),
            "sand_total": round(sum(st.sand.values()), 4),
            "stalled": _round(st.belt_stalled, 1),
            "tappable": [list(p) for p in st.tappable],
            "grid": res.view.labels() if res.view is not None else None,
            "locks": [f"{l.kind}{'' if l.counter is None else l.counter}" for l in st.locks]})
    if d is not None:
        rec.update({"reason": d.reason,
                    "move": candidate_record(d.move) if d.move is not None else None,
                    "candidates": [candidate_record(c) for c in d.candidates]})
    return rec


def git_commit() -> Optional[str]:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                           text=True, timeout=5, cwd=Path(__file__).resolve().parent)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


class RunLog:
    def __init__(self, root: Path, cfg: Any, args: Dict[str, Any], max_frames: int = 1500,
                 clock=time.monotonic, stamp: Optional[str] = None) -> None:
        stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
        self.dir = Path(root) / "runs" / stamp
        n = 1
        while self.dir.exists():   # two runs in one second
            n += 1
            self.dir = Path(root) / "runs" / f"{stamp}-{n}"
        (self.dir / "frames").mkdir(parents=True)
        self._clock = clock
        self.t0 = clock()
        self.max_frames = max_frames
        self.frames_saved = 0
        self._events = (self.dir / "events.jsonl").open("a", encoding="utf-8")
        self._log = (self.dir / "bot.log").open("a", encoding="utf-8")
        self._belt_fh = (self.dir / "belt.csv").open("a", newline="", encoding="utf-8")
        self._belt = csv.writer(self._belt_fh)
        self._belt.writerow(["t", "frame", "k", "slots", "mean_fill"])
        cfg_data = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else None
        self._write_json("run.json", {"started": time.strftime("%Y-%m-%d %H:%M:%S"),
                                      "args": args, "commit": git_commit(), "config": cfg_data})

    def t(self) -> float:
        return round(self._clock() - self.t0, 3)

    def _write_json(self, name: str, data: Any) -> None:
        (self.dir / name).write_text(json.dumps(data, indent=1, default=str) + "\n",
                                     encoding="utf-8")

    def event(self, kind: str, **data: Any) -> Dict[str, Any]:
        rec = {"type": kind, "t": self.t(), **data}
        self._events.write(json.dumps(rec, default=str) + "\n")
        self._events.flush()
        return rec

    def log(self, msg: str) -> None:
        line = f"[{self.t():8.2f}] {msg}"
        print(line)
        self._log.write(line + "\n")
        self._log.flush()

    def belt(self, frame: int, res: "Analysis") -> None:
        st = res.state
        if st is None:
            return
        slots = "|".join(f"{b.color}{'/' + '/'.join(b.stages) if b.stages else ''}"
                         f" {'?' if b.fill is None else b.fill}" for b in st.conveyor)
        fills = [b.fill for b in st.conveyor if b.fill is not None]
        mean = round(sum(fills) / len(fills), 1) if fills else ""
        self._belt.writerow([self.t(), frame, "" if st.conveyor_k is None else st.conveyor_k,
                             slots, mean])
        self._belt_fh.flush()

    def frame(self, name: str, img: "np.ndarray") -> Optional[str]:
        """Save a PNG under frames/ (capped per run); returns its path relative to the run."""
        import cv2

        if self.frames_saved >= self.max_frames:
            return None
        rel = f"frames/{name}.png"
        if not cv2.imwrite(str(self.dir / rel), img):
            self.log(f"warning: cannot write {rel}")
            return None
        self.frames_saved += 1
        return rel

    def result(self, result: str, reason: str, **data: Any) -> None:
        rec = {"result": result, "reason": reason, "duration_s": self.t(), **data}
        self.event("stop", **rec)
        self._write_json("result.json", rec)

    def close(self) -> None:
        for fh in (self._events, self._log, self._belt_fh):
            fh.close()


def load_events(run_dir: Path) -> List[Dict[str, Any]]:
    out = []
    with (Path(run_dir) / "events.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
