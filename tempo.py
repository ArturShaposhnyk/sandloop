"""Tempo for bot.py (pure logic, no OpenCV): which colours are about to be covered, releasing
several buckets back to back from one confirmed decision, and what the bot remembers of the
towers it released.

* ``ExposureTrend``: a colour whose contact-layer share dropped by ``cover_min_drop`` within
  ``cover_window_s`` while none of its buckets is on the belt (so the drop is not our own
  collecting) is being covered by falling sand: it gets ``weight_covering`` x drop on its score
  (``prioritize``). A heuristic of the bot, not a game rule.
* ``burst``: after the confirmed move, the next taps are chosen by ``strategy.choose`` on the
  state simulated after the previous taps (belt demand, free slots, sufficiency all updated).
  A follow-up must be tappable in the real frames (not a bucket the earlier taps only open:
  it still has to flip), never takes the last free slot and its colour must be exposed now.
* ``ReleaseLedger``: buckets / towers the bot tapped. Guards against tapping a cell twice while
  its bucket is still flying, and gives a tower on the belt its later stages when the belt
  reader can't see them (the belt shows the collecting stage and at most the next one).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from config import BotConfig, StrategyConfig
from state import LevelState, Pos
from strategy import Candidate, Decision, choose


class ExposureTrend:
    """Contact-layer shares over the last ``window_s`` seconds."""

    def __init__(self, window_s: float, min_drop: float) -> None:
        self.window_s, self.min_drop = window_s, min_drop
        self._hist: Deque[Tuple[float, Dict[str, float]]] = deque()

    def update(self, now: float, exposed: Dict[str, float]) -> None:
        self._hist.append((now, dict(exposed)))
        while self._hist and now - self._hist[0][0] > self.window_s:
            self._hist.popleft()

    def covering(self, on_belt: Iterable[str] = ()) -> Dict[str, float]:
        """colour -> drop of its share since the window start (only drops >= min_drop, only
        colours with no bucket on the belt)."""
        if len(self._hist) < 2:
            return {}
        belt = set(on_belt)
        (_, old), (_, new) = self._hist[0], self._hist[-1]
        out = {}
        for c, was in old.items():
            drop = was - new.get(c, 0.0)
            if c not in belt and drop >= self.min_drop:
                out[c] = drop
        return out

    def reset(self) -> None:
        self._hist.clear()


def _rank(c: Candidate) -> tuple:
    """strategy.choose's order: score, colour not on the belt, front row, left."""
    return (round(c.score, 6), not c.on_belt, -c.pos[0], -c.pos[1])


def prioritize(decision: Decision, covering: Dict[str, float], weight: float) -> Decision:
    """Add the cover bonus to allowed candidates of a covered colour and re-pick the best."""
    if decision.move is None or not covering or weight <= 0 or decision.move.level_plan:
        return decision   # a level-plan step is the plan's, not re-picked
    bonus = {}
    for cand in decision.candidates:
        if cand.rejected is None and cand.color in covering:
            bonus[cand.pos] = weight * covering[cand.color]
            cand.score += bonus[cand.pos]
    if not bonus:
        return decision
    best = max((c for c in decision.candidates if c.rejected is None), key=_rank)
    r, c = best.pos
    reason = f"tap r{r}c{c} {best.color} (score {best.score:.2f})"
    if best.pos in bonus:
        reason += f", being covered (+{bonus[best.pos]:.2f})"
    if best.plan is not None:
        reason += f", {best.plan.describe(1, detail=False)}"
    return replace(decision, move=best, reason=reason)


def burst(state: LevelState, first: Candidate, real_tappable: Set[Pos], sc: StrategyConfig,
          bc: BotConfig, covering: Optional[Dict[str, float]] = None
          ) -> List[Tuple[Candidate, str]]:
    """The confirmed move plus up to ``burst_max - 1`` follow-ups: [(candidate, reason)]."""
    moves = [(first, "confirmed")]
    sim = state
    while len(moves) < bc.burst_max:
        sim = sim.after_tap(moves[-1][0].pos)
        free = sim.free_slots
        if free is None or free < bc.burst_min_free:
            break
        taken = {m.pos for m, _ in moves}
        sim.tappable = [p for p in sim.tappable if p in real_tappable and p not in taken]
        if not sim.tappable:
            break
        d = prioritize(choose(sim, sc), covering or {}, bc.weight_covering)
        if (d.move is None or d.move.avail < bc.burst_min_avail
                or d.move.score < bc.burst_min_score or d.move.look_bonus < 0
                or d.move.gamble or d.move.sacrifice or d.move.lost):
            break   # a follow-up must be a clearly good move by the same checks (run 145310)
        moves.append((d.move, f"burst {len(moves) + 1}: {d.reason}"))
    return moves


# --- released buckets / towers ---------------------------------------------------------------


@dataclass
class Released:
    time: float
    pos: Pos
    stages: Tuple[str, ...]             # top first (a plain bucket: one stage)
    current: int = 0                    # index of the stage collecting on the belt
    last_seen: float = 0.0

    @property
    def tower(self) -> bool:
        return len(self.stages) > 1

    @property
    def color(self) -> str:
        return self.stages[self.current]

    def remaining(self) -> Tuple[str, ...]:
        return self.stages[self.current + 1:]


@dataclass
class ReleaseLedger:
    retap_guard_s: float = 2.0
    forget_s: float = 3.0               # tower no longer seen on the belt this long = done
    released: List[Released] = field(default_factory=list)

    def add(self, now: float, pos: Pos, color: str, body: Optional[str]) -> Released:
        rel = Released(now, pos, (color,) + ((body,) if body else ()), last_seen=now)
        self.released.append(rel)
        return rel

    def guarded(self, pos: Pos, now: float) -> bool:
        """True while a cell we tapped may still show its (flying) bucket."""
        return any(r.pos == pos and now - r.time < self.retap_guard_s for r in self.released)

    def towers(self) -> List[Released]:
        return [r for r in self.released if r.tower and r.current < len(r.stages)]

    def update(self, state: LevelState, now: float) -> None:
        """Follow released towers on the belt and fill in stages the belt reader missed:
        the current stage left (100 %) and the next one shows -> advance; a belt slot of the
        tower's colour with no stages read gets the remaining ones (only if unambiguous)."""
        colours = [b.color for b in state.conveyor]
        for t in self.towers():
            if t.color in colours:
                t.last_seen = now
            elif t.remaining() and t.remaining()[0] in colours:
                t.current += 1
                t.last_seen = now
            elif now - t.last_seen > self.forget_s:
                t.current = len(t.stages)   # done (or not followable any more)
        live = [t for t in self.towers() if t.remaining()]
        for i, b in enumerate(state.conveyor):
            if b.stages:
                continue
            match = [t for t in live if t.color == b.color]
            same = [x for x in state.conveyor if x.color == b.color]
            if len(match) == 1 and len(same) == 1:
                state.conveyor[i] = replace(b, stages=match[0].remaining())

    def describe(self) -> Sequence[str]:
        return [f"r{r.pos[0]}c{r.pos[1]} {'/'.join(r.stages)}"
                + (f" (stage {r.current + 1})" if r.tower else "") for r in self.released]
