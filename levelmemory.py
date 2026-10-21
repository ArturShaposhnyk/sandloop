"""Level memory (docs/GAME_RULES.md "Level plan"): earlier attempts of the same board from
<debug>/levels.jsonl (bot.py writes one line per level; Try Again = same board, levels are
deterministic).

* ``banned``: tap sequences of attempts that did not win - the planner never repeats one whole.
  Run 20261021-101655 (L189, 3 losses): the check compared the plan's OWN taps (counted from the
  re-plan's board) with whole-level sequences, so nothing was ever banned after the first
  re-plan; the planner now prefixes the level's taps so far (``PlanRunner.done``).
* ``revealed``: colours seen under ice / key locks / curtains in earlier attempts (a lock
  covers buckets whose colour shows only once it opens). The simulator uses them instead of
  "colour unknown" (the plan's horizon), so a plan can reach past the ice on Try Again.
* ``capacity_shares``: the picture share one bucket stage of a colour really took.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

Pos = Tuple[int, int]


def level_signature(labels: Sequence[str]) -> str:
    """Short id of a board (the grid labels of a level's first frame) without the buckets'
    faces: they follow from the path rule, and a first frame still fading in after an ad read
    the top row face-down (run 20261021-101655 attempt 3: its own signature, no memory)."""
    norm = [" ".join(lab.rsplit("-", 1)[0] if lab.endswith(("-up", "-down")) else lab
                     for lab in row.split()) for row in labels]
    return hashlib.sha1("\n".join(norm).encode()).hexdigest()[:12]


def _record_signature(r: Dict[str, Any]) -> Optional[str]:
    """A record's board signature, from its start grid (records from before it was stored, run
    20261020-145310: the grid of the decision behind the first tap); else the stored one."""
    taps = r.get("tap_log") or []
    grid = (r.get("start") or {}).get("grid") or \
        ((taps[0].get("decision") or {}).get("grid") if taps else None)
    return level_signature(grid) if grid else r.get("signature")


HIDING = ("ice", "lock:", "cur")   # start labels of cells that cover a bucket of unseen colour


def _colours(labels: Sequence[str]) -> Set[str]:
    out: Set[str] = set()
    for row in labels:
        for lab in row.split():
            head = lab.split("-")[0]
            if (head[:1].isalpha() and lab != "x" and ":" not in head
                    and not lab.startswith(HIDING)):
                out.update(head.split("/"))
    return out


def _clean_end(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The attempt's last board read, unless it was read under a popup (run 20261021-101655:
    the revive popup sliding in turned the board "green" and the sand to ~0 - before
    ``screens.popup_bar``): a bucket or sand colour the level's start shows nowhere."""
    taps = r.get("tap_log") or []
    end = r.get("end") or {}
    start = r.get("start") or ((taps[0].get("decision") or {}) if taps else {})
    if not end.get("grid") or not start.get("grid"):
        return end or None
    known = _colours(start["grid"]) | set(start.get("sand") or {}) | {"mystery"}
    seen = _colours(end["grid"]) | set(end.get("sand") or {})
    return end if seen <= known else None


class LevelMemory:
    """Earlier attempts of a board from <debug>/levels.jsonl (bot.py writes one line per level)."""

    def __init__(self, path: Path) -> None:
        self.records: List[Dict[str, Any]] = []
        if Path(path).exists():
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                try:
                    self.records.append(json.loads(line))
                except ValueError:
                    continue

    def attempts(self, signature: str) -> List[Dict[str, Any]]:
        return [r for r in self.records if _record_signature(r) == signature]

    def banned(self, signature: str) -> Set[Tuple[Pos, ...]]:
        """Whole-level tap sequences of attempts that did not win: a plan must not repeat one of
        them whole (it may share a prefix and deviate later)."""
        out = set()
        for r in self.attempts(signature):
            if r.get("result") != "win" and r.get("tap_log"):
                out.add(tuple(tuple(t["pos"]) for t in r["tap_log"]))
        return out

    def revealed(self, signature: str) -> Dict[Pos, Tuple[str, Optional[str]]]:
        """Cell -> (colour, lower stage or None) seen under ice / locks / curtains in earlier
        attempts of the board (the latest attempt wins)."""
        out: Dict[Pos, Tuple[str, Optional[str]]] = {}
        for r in self.attempts(signature):
            seen = dict(r.get("revealed") or {})
            end, start = _clean_end(r), (r.get("start") or {}).get("grid")
            if end is not None and start and end.get("grid") and not seen:   # records before "revealed": start vs end grid
                for i, (a, b) in enumerate(zip(start, end["grid"])):
                    for j, (la, lb) in enumerate(zip(a.split(), b.split())):
                        if la.startswith(HIDING) and lb.endswith(("-up", "-down")):
                            seen[f"{i},{j}"] = lb.rsplit("-", 1)[0]
            for key, label in seen.items():
                row, col = (int(x) for x in key.split(","))
                top, _, body = label.partition("/")
                out[(row, col)] = (top, body or None)
        return out

    def capacity_shares(self, signature: str) -> Dict[str, float]:
        """Per colour, the picture share one bucket stage really took in earlier attempts:
        its sand drop / its stages filled (stages tapped minus what the belt still needed of it
        at the end). Corrects the simulator's balanced guess where it was wrong."""
        drop: Dict[str, float] = {}
        filled: Dict[str, float] = {}
        for r in self.attempts(signature):
            taps = r.get("tap_log") or []
            start = r.get("start") or ((taps[0].get("decision") or {}) if taps else {})
            end = _clean_end(r) or {}
            if not start.get("sand") or not end.get("sand"):
                continue
            done: Dict[str, float] = {}
            for t in r.get("tap_log") or []:
                for col in [t.get("color")] + list(t.get("stages") or []):
                    if col:
                        done[col] = done.get(col, 0.0) + 1
            for b in end.get("conveyor") or []:
                if b.get("color"):
                    fill = b["fill"] if b.get("fill") is not None else 50
                    done[b["color"]] = done.get(b["color"], 0.0) - (1 - fill / 100)
                for col in b.get("stages") or []:
                    done[col] = done.get(col, 0.0) - 1
            for col, n in done.items():
                d = start["sand"].get(col, 0.0) - end["sand"].get(col, 0.0)
                if n >= 1 and d > 0:
                    drop[col] = drop.get(col, 0.0) + d
                    filled[col] = filled.get(col, 0.0) + n
        return {c: drop[c] / filled[c] for c in drop}
