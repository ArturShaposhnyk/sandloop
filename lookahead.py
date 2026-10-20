"""Phase 7: look 2-3 taps ahead before a tap (docs/GAME_RULES.md, "Lookahead").

Pure logic on ``state.LevelState``. The question per candidate tap: after it (and the best 1-2
follow-up taps the rules allow), how many belt slots are still **usable** - free, or holding a
bucket that will get sand - and does the belt **jam** (full, nothing on it can fill)?

Sand model (coarse, one step): the belt buckets that collect now take their colour's contact
layer; the sand right above it falls down (``LevelState.above``, from sand.contact_layers). So
releasing a colour that covers the bottom (L187: the purple sky) uncovers the colours stalled
on the belt. A belt bucket is **live** when

  * enough of its colour is left to fill what the belt still needs of it (``look_sand_margin``
    x need x capacity; stricter than the release rule's margin), and
  * its colour is in the contact layer (``look_min_exposed``) and not stalled on the belt now,
    or the settle step uncovers at least ``look_min_gain`` more of it.

Value of a state = (free slots + live belt buckets) / slots. A candidate's outlook is the best
value over its continuations of up to ``look_depth`` taps (stopping early is allowed: the bot
may wait), each follow-up chosen among the ``look_width`` best-exposed buckets that pass the
release rules there. ``strategy.choose`` adds ``weight_lookahead`` x (outlook - value now) to
the score and rejects a tap that leaves the belt jammed (full, nothing live).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from config import StrategyConfig
from state import LevelState, Pos

# why releasing the bucket at pos in this state breaks the rules (None = allowed)
Check = Callable[[LevelState, Pos], Optional[str]]


@dataclass
class BeltOutlook:
    free: int
    live: int
    dead: int
    slots: int
    contact: Dict[str, float] = field(default_factory=dict)   # after the settle step

    @property
    def value(self) -> float:
        return (self.free + self.live) / self.slots if self.slots else 0.0

    @property
    def jam(self) -> bool:
        return self.free == 0 and self.live == 0

    def describe(self) -> str:
        return (f"{self.live} live / {self.dead} dead on the belt, {self.free} free"
                + (" - JAM" if self.jam else ""))


@dataclass
class Outlook:
    steps: List[Tuple[Pos, str]]          # the best continuation, the candidate first
    belt: BeltOutlook

    @property
    def value(self) -> float:
        return self.belt.value

    def describe(self) -> str:
        path = " -> ".join(f"r{r}c{c} {col}" for (r, c), col in self.steps)
        return f"lookahead {path}: {self.belt.describe()}"


def _enough_sand(state: LevelState, colour: str, sc: StrategyConfig) -> bool:
    cap = state.capacity_estimate()
    if not cap:
        return True
    return state.sand.get(colour, 0.0) >= state.conveyor_need(colour) * cap * sc.look_sand_margin


def _collecting(state: LevelState, colour: str, sc: StrategyConfig) -> bool:
    """The belt's ``colour`` buckets collect now: exposed, not stalled, sand left."""
    stalled = state.belt_stalled.get(colour, 0.0) >= sc.stall_s
    return (not stalled and state.exposed.get(colour, 0.0) >= sc.look_min_exposed
            and _enough_sand(state, colour, sc))


def settle(state: LevelState, sc: StrategyConfig) -> Dict[str, float]:
    """The contact layer after the collecting belt colours took theirs once and the sand above
    fell down (``state.above``; a colour without a row there keeps its share)."""
    out = dict(state.exposed)
    for colour in {b.color for b in state.conveyor if b.color and (b.fill or 0) < 100}:
        if not _collecting(state, colour, sc) or colour not in state.above:
            continue
        out[colour] = out.get(colour, 0.0) - state.exposed.get(colour, 0.0)
        for d, share in state.above[colour].items():
            out[d] = out.get(d, 0.0) + share
    return {c: max(0.0, s) for c, s in out.items()}


def belt_outlook(state: LevelState, sc: StrategyConfig) -> BeltOutlook:
    contact = settle(state, sc)
    live = dead = 0
    for b in state.conveyor:
        if b.fill is not None and b.fill >= 100:
            continue
        c = b.color
        if c is None:                      # not read: assume it collects
            live += 1
            continue
        gain = contact.get(c, 0.0) - state.exposed.get(c, 0.0)
        ok = _enough_sand(state, c, sc) and (_collecting(state, c, sc)
                                             or gain >= sc.look_min_gain)
        live, dead = (live + 1, dead) if ok else (live, dead + 1)
    free = state.free_slots or 0
    return BeltOutlook(free, live, dead, state.conveyor_slots, contact)


def outlook(state: LevelState, pos: Pos, check: Check, sc: StrategyConfig) -> Outlook:
    """Best continuation of up to ``look_depth`` taps starting with ``pos`` (allowed by the
    caller already)."""
    first = (pos, state.cell(pos).color or "?")
    best = Outlook([first], belt_outlook(state.after_tap(pos), sc))

    def walk(st: LevelState, steps: List[Tuple[Pos, str]]) -> None:
        nonlocal best
        here = belt_outlook(st, sc)
        if here.value > best.value + 1e-9:
            best = Outlook(list(steps), here)
        if len(steps) >= sc.look_depth or not st.free_slots:
            return
        nxt = [p for p in st.tappable if check(st, p) is None]
        nxt.sort(key=lambda p: -st.exposed.get(st.cell(p).color or "", 0.0))
        for p in nxt[:sc.look_width]:
            walk(st.after_tap(p), steps + [(p, st.cell(p).color or "?")])

    walk(state.after_tap(pos), [first])
    return best
