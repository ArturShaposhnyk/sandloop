"""Whole-level planning (docs/GAME_RULES.md "Level plan"; user, 2026-10-20).

The whole board is known at level start (a face-down bucket shows its real colour; Try Again
restarts the same level), so the bot searches tap / wait sequences in ``sim.Sim`` and picks one
that never jams the belt:

* **Beam search** (``plan_level``): each depth expands every node by its taps and a wait,
  keeps the ``beam`` best by score = sand collected + win bonus - dead belt buckets (their
  colour has not enough sand left) + free slots; jammed nodes are dropped; sequences a
  ``LevelMemory`` banned (the attempt that jammed last time on this board) are pruned. Stops at
  a win, when nothing can move, at ``max_depth`` or after ``budget_s``.
* **Following it live** (``PlanRunner``): the plan records the board it expects before each
  tap, the belt (colours, fill %) and the belt time. A frame whose board matches step i taps
  that step's bucket once the live belt has no more buckets than the plan had there (else it
  waits, as the plan did). A board that matches no step (a bucket filled earlier / later, an
  opened lock showed its colours, a misread) is re-planned from that frame. When the plan has
  no step left (unknown buckets ahead, horizon) the greedy strategy decides.
* **A wait never stalls** (run 161536: 32 s on "k 3 > 2" with nothing collecting): the wait
  is given up when the belt is ``wait_grace_s`` behind the plan's predicted time or no belt
  bucket gained sand for ``wait_stall_s``. Then the bot re-plans from the frame (the real
  belt); if that plan waits as well, the greedy strategy decides until the board changes.
  Every tap logs the plan's predicted belt and time next to the real ones ("plan check").
* **Level memory** (``LevelMemory``, debug/levels.jsonl): attempts of the same board (same
  start signature) that did not win ban their tap sequence; their sand use calibrates the
  bucket capacity of the simulator.

    python -m levelplan dataset/screens/*-064-*.png   # the plan for a start frame (no taps)
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from sim import Picture, Sig, Sim, SimConfig, grid_signature, picture_from_frame
from state import LevelState, Pos


def level_signature(labels: Sequence[str]) -> str:
    """Short id of a board (the grid labels of a level's first frame)."""
    return hashlib.sha1("\n".join(labels).encode()).hexdigest()[:12]


@dataclass
class Step:
    pos: Pos
    colour: str
    belt_before: int                  # buckets on the belt in the plan right before the tap
    board: Sig                        # board the plan expects before the tap
    t: float = 0.0                    # predicted belt seconds since the plan was made
    belt: List[Tuple[str, int]] = field(default_factory=list)   # predicted (colour, fill %)


@dataclass
class Plan:
    steps: List[Step]
    result: str                       # "win" | "horizon" (unknown buckets / nothing to do) | "open"
    score: float
    collected: float                  # share of the picture's sand collected by the end
    dead: int                         # dead belt buckets at the end
    nodes: int = 0
    seconds: float = 0.0
    actions: List[str] = field(default_factory=list)

    def describe(self) -> List[str]:
        head = (f"level plan: {len(self.steps)} taps, {self.result}, collects "
                f"{self.collected:.0%} of the sand, {self.dead} dead on the belt at the end "
                f"({self.nodes} nodes, {self.seconds:.1f}s)")
        return [head] + [f"  {a}" for a in self.actions]


def _score(s: Sim, total: int, pc: Any) -> float:
    return (s.collected / max(1, total) + (10.0 if s.won else 0.0)
            - pc.weight_dead * s.dead_slots() - pc.weight_stuck * s.stuck_slots()
            + pc.weight_free * s.free - pc.weight_full * s.full_taps)


def sim_config(pc: Any, slots: int) -> SimConfig:
    return SimConfig(slots, pc.lap_s, pc.fill_rate, pc.fill_min_exposed, pc.fill_max, pc.tap_s,
                     pc.max_wait_s)


def plan_level(state: LevelState, pic: Picture, pc: Any,
               banned: Set[Tuple[Pos, ...]] = frozenset(),
               capacity: Optional[Dict[str, int]] = None) -> Plan:
    """Beam search from ``state`` (+ its picture); ``pc``: PlannerConfig."""
    t0 = time.monotonic()
    root = Sim.start(state, pic, sim_config(pc, state.conveyor_slots), capacity)
    root.grid.disagreements = []
    root.grid.tappable = root.grid.playable()
    total = max(1, pic.total())
    beam, best, nodes = [root], root, 0
    for _ in range(pc.max_depth):
        children: List[Sim] = []
        for s in beam:
            nxt = [s.tap(p) for p in s.moves() if tuple(s.taps + [p]) not in banned]
            if s.belt:
                w = s.wait()
                if w is not None:
                    nxt.append(w)
            nodes += len(nxt)
            children += [c for c in nxt if not c.jam]
        if not children:
            break
        seen, uniq = set(), []
        for c in sorted(children, key=lambda c: -_score(c, total, pc)):
            key = c.signature()           # one node per board: keeps the beam diverse
            if key not in seen:
                seen.add(key)
                uniq.append(c)
        beam = uniq[:pc.beam]
        if _score(beam[0], total, pc) > _score(best, total, pc):
            best = beam[0]
        if best.won or time.monotonic() - t0 > pc.budget_s:
            break
    return _plan_of(root, best, total, nodes, time.monotonic() - t0, pc)


def _plan_of(root: Sim, best: Sim, total: int, nodes: int, seconds: float, pc: Any) -> Plan:
    """Replay ``best``'s actions from ``root`` to record the board before every tap."""
    s, steps, lines = root, [], []
    for kind, pos, colour, belt in best.actions:
        if kind == "tap":
            steps.append(Step(pos, colour, len(s.belt), s.signature(), s.t, s.belt_fills()))
            at = s.t
            s = s.tap(pos)
            lines.append(f"{len(steps):2d}. {at:4.0f}s tap r{pos[0]}c{pos[1]} {colour}  (belt "
                         f"{belt} -> {len(s.belt)}: {_belt_text(s.belt_fills())})")
        else:
            s = s.wait() or s
            lines.append(f"    {s.t:4.0f}s wait (belt {len(s.belt)}: "
                         f"{_belt_text(s.belt_fills())}; sand left {s.pic.total() / total:.0%})")
    result = "win" if best.won else ("horizon" if best.unknown_left() else "open")
    return Plan(steps, result, _score(best, total, pc), best.collected / total, best.dead_slots(),
                nodes, seconds, lines)


def _belt_text(fills: Sequence[Tuple[str, Optional[int]]]) -> str:
    return ", ".join(f"{c} {'?' if f is None else f}" for c, f in fills) or "empty"


# --- level memory ------------------------------------------------------------------------------


def _record_signature(r: Dict[str, Any]) -> Optional[str]:
    """A record's board signature; records from before it was stored (run 20261020-145310):
    the grid of the decision behind the first tap (the level's first frames)."""
    if r.get("signature"):
        return r["signature"]
    taps = r.get("tap_log") or []
    grid = ((taps[0].get("decision") or {}).get("grid") if taps else None) or \
        (r.get("start") or {}).get("grid")
    return level_signature(grid) if grid else None


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
        """Tap sequences of attempts that did not win: the plan must not repeat one of them
        whole (it may share a prefix and deviate later)."""
        out = set()
        for r in self.attempts(signature):
            if r.get("result") != "win" and r.get("tap_log"):
                out.add(tuple(tuple(t["pos"]) for t in r["tap_log"]))
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
            end = r.get("end") or {}
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


# --- following the plan live -------------------------------------------------------------------


class PlanRunner:
    """Per level: plans on the first frame, follows the plan, re-plans when the board differs."""

    def __init__(self, cfg: Any, memory_path: Optional[Path] = None) -> None:
        self.cfg, self.pc = cfg, cfg.planner
        self.memory_path = memory_path
        self.reset()

    def reset(self) -> None:
        self.plan: Optional[Plan] = None
        self.signature: Optional[str] = None
        self.replans = 0
        self.last_plan_t = -1e9
        self.log: List[Dict[str, Any]] = []     # plans made this level (levels.jsonl)
        self.at: Optional[int] = None           # step whose board the frames show
        self.at_since = 0.0                     # ... since
        self.checked: Set[int] = set()          # steps whose tap was compared with the plan
        self.given_up: Optional[Sig] = None     # board on which the strategy decides

    def decide(self, img, gv, state: LevelState, fallback, now: float, log=print):
        """The decision for this frame: the plan's step, a wait for it, or ``fallback``."""
        from strategy import Decision, _score as score_cand, evaluate

        if self.signature is None:
            self.signature = level_signature(gv.labels())
        board = grid_signature(state.grid)
        if board == self.given_up:
            return fallback
        self.given_up = None
        step_i = self._step(board)
        if step_i is None and now - self.last_plan_t >= self.pc.replan_min_s:
            self._replan(img, state, now, log, "start" if self.plan is None else "board differs")
            step_i = self._step(board)
        if step_i is None:
            return fallback
        if step_i != self.at:
            self.at, self.at_since = step_i, now
        step = self.plan.steps[step_i]
        n = len(self.plan.steps)
        k = state.conveyor_k
        if k is None or state.free_slots == 0 or k > step.belt_before:
            why = self._stalled(state, step_i, now)
            if why is None:
                return Decision(None, f"wait: level plan step {step_i + 1}/{n} (tap r{step.pos[0]}"
                                      f"c{step.pos[1]} {step.colour}) waits for the belt (k {k} > "
                                      f"{step.belt_before})", fallback.candidates)
            return self._give_up(img, state, board, fallback, now, log, why)
        if step.pos not in state.tappable:
            return Decision(None, f"wait: level plan step {step_i + 1}/{n}: r{step.pos[0]}"
                                  f"c{step.pos[1]} not tappable yet", fallback.candidates)
        self._check(state, step_i, now, log)
        cand = evaluate(state, step.pos, self.cfg.strategy, gamble=True)
        cand.rejected = None
        score_cand(state, cand, self.cfg.strategy)
        cand.level_plan = f"{step_i + 1}/{n}"
        return Decision(cand, f"tap r{step.pos[0]}c{step.pos[1]} {step.colour} (level plan step "
                              f"{step_i + 1}/{n}, {self.plan.result})", fallback.candidates)

    def _step(self, board: Sig) -> Optional[int]:
        if self.plan is None:
            return None
        for i, s in enumerate(self.plan.steps):
            if s.board == board:
                return i
        return None

    def predicted_wait(self, i: int) -> float:
        """Belt seconds the plan expects between the board of step ``i`` and its tap."""
        steps = self.plan.steps
        before = steps[i - 1].t + self.pc.tap_s if i > 0 else 0.0
        return max(0.0, steps[i].t - before)

    def _stalled(self, state: LevelState, i: int, now: float) -> Optional[str]:
        """Why the wait for step ``i`` is given up, or None (keep waiting)."""
        waited = now - self.at_since
        expect = self.predicted_wait(i)
        clocks = [state.belt_stalled[b.color] for b in state.conveyor
                  if b.color in state.belt_stalled and (b.fill is None or b.fill < 100)]
        idle = min(clocks) if clocks else None
        if idle is not None and idle >= self.pc.wait_stall_s and waited >= self.pc.wait_stall_s:
            return (f"no belt bucket gained sand for {idle:.0f}s (waited {waited:.0f}s, plan "
                    f"{expect:.0f}s)")
        if waited > expect + self.pc.wait_grace_s:
            return f"the belt is behind the plan (waited {waited:.0f}s, plan {expect:.0f}s)"
        return None

    def _give_up(self, img, state: LevelState, board: Sig, fallback, now: float, log, why: str):
        """Leave a stalled wait: re-plan from the real belt; if that plan waits too, the
        strategy decides until the board changes."""
        i, n = self.at, len(self.plan.steps)
        self._check(state, i, now, log, gave_up=why)
        log(f"level plan step {i + 1}/{n} gives up waiting: {why} - re-planning")
        self._replan(img, state, now, log, "wait stalled")
        j = self._step(board)
        k = state.conveyor_k
        if j is None or k is None or k > self.plan.steps[j].belt_before:
            self.given_up = board
            log("level plan: the new plan waits as well - the strategy decides until the board "
                "changes")
            return fallback
        self.at, self.at_since = j, now
        return self.decide(img, None, state, fallback, now, log)

    def _check(self, state: LevelState, i: int, now: float, log, gave_up: str = "") -> None:
        """Log the plan's belt / time before step ``i`` next to the real ones (once per step)."""
        if i in self.checked or not self.log:
            return
        self.checked.add(i)
        step, rec = self.plan.steps[i], self.log[-1]
        real = [(b.color or "?", b.fill) for b in state.conveyor]
        t_real = now - rec["t"]
        log(f"plan check step {i + 1}/{len(self.plan.steps)}: t {t_real:.0f}s (plan "
            f"{step.t:.0f}s), belt k {state.conveyor_k} (plan {step.belt_before}): plan "
            f"{_belt_text(step.belt)} | real {_belt_text(real)}")
        rec.setdefault("checks", []).append(
            {"step": i + 1, "t": round(t_real, 1), "t_plan": round(step.t, 1),
             "k": state.conveyor_k, "k_plan": step.belt_before,
             "belt": [[c, f] for c, f in real], "belt_plan": [[c, f] for c, f in step.belt],
             **({"gave_up": gave_up} if gave_up else {})})

    def _replan(self, img, state: LevelState, now: float, log, why: str) -> None:
        mem = LevelMemory(self.memory_path) if self.memory_path else None
        banned = mem.banned(self.signature) if mem else set()
        pic = picture_from_frame(img, _palette(self.cfg), self.cfg.vision, self.pc.cell_px)
        cap = None
        shares = mem.capacity_shares(self.signature) if mem else {}
        sand = sum(state.sand.values())
        if shares and sand > 0:   # picture share -> cells, at this frame's cells per share
            cap = {c: max(1, round(v * pic.total() / sand)) for c, v in shares.items()}
        self.plan = plan_level(state, pic, self.pc, banned, cap)
        self.last_plan_t = now
        self.at, self.checked = None, set()
        self.replans += why != "start"
        self.log.append({"t": round(now, 2), "why": why, "taps": len(self.plan.steps),
                         "result": self.plan.result, "collected": round(self.plan.collected, 3),
                         "banned": len(banned), "capacity": cap,
                         "steps": [[*s.pos, s.colour, round(s.t, 1)] for s in self.plan.steps]})
        log(f"level plan ({why}{', ' + str(len(banned)) + ' failed sequence(s) avoided' if banned else ''}"
            f"{', capacity from earlier attempts' if cap else ''}):")
        for line in self.plan.describe():
            log(line)


_PALETTES: Dict[str, Any] = {}


def _palette(cfg):
    from palette import Palette

    if cfg.paths.palette not in _PALETTES:
        _PALETTES[cfg.paths.palette] = Palette.load(cfg.paths.palette)
    return _PALETTES[cfg.paths.palette]


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    import cv2

    from analyzer import Analyzer
    from config import load_config

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("frames", nargs="+", help="gameplay frames (a level's first frame)")
    ap.add_argument("--memory", help="levels.jsonl with earlier attempts (default: <debug>/)")
    args = ap.parse_args(argv)
    cfg = load_config()
    an = Analyzer(cfg)
    for path in args.frames:
        img = cv2.imread(path)
        res = an.analyze(img, log=lambda s: None)
        if res.state is None:
            print(f"{path}: not a readable gameplay frame ({res.screen}, {res.failure})")
            continue
        runner = PlanRunner(cfg, Path(args.memory) if args.memory
                            else Path(cfg.paths.debug) / "levels.jsonl")
        print(f"== {path}")
        print("\n".join(res.view.labels()))
        runner.decide(img, res.view, res.state, res.decision, 0.0, log=print)
        out = Path(cfg.paths.debug) / f"levelplan-{Path(path).stem}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), draw_plan(img, res.view, runner.plan))
        print(f"-> {out}")
    return 0


def draw_plan(img, gv, plan: Optional[Plan]):
    """The frame with every planned tap numbered on its cell."""
    import cv2

    out = img.copy()
    if plan is None:
        return out
    for i, s in enumerate(plan.steps, 1):
        x0, y0, x1, y1 = gv.lattice.rect(*s.pos)
        cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
        cv2.circle(out, (cx, cy), 13, (0, 0, 0), -1)
        cv2.putText(out, str(i), (cx - 9 if i > 9 else cx - 5, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
