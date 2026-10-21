"""Belief of the belt (redesign phase C, decider.py): every belt slot tracked by identity across
frames, with its observed fill and fill rate.

* **Slots** - the k/5 counter is authoritative for how many there are (BeltTracker's lesson,
  run 20261021-122825): a tap adds a slot with what the bot knows of it (colour, tower stages,
  fill 0); the counter falling removes the most likely finished one (read at 100 %, else the
  fullest one not read this frame); a counter above the slots known adds a slot of an unread
  colour. A fresh slot is never dropped for a counter that has not caught up with the tap yet.
* **Readings -> slots** - the belt is a loop and only part of it is on screen, so a frame
  shows some slots, in no fixed order. Per colour each read fill is matched to the slot whose
  last fill is the highest at or below it (+1 for label jitter; fills only grow: two buckets of
  a colour read alternately are no growth). A reading at 100 % with a later tower stage moves
  that slot to its next stage; a tower whose next stage's colour shows is advanced too.
* **Gaining / stalled** - a slot is *gaining* when it grew by ``gain_min`` points within
  ``gain_window_s`` (or was tapped less than ``fresh_grace_s`` ago); *stalled* when it was read
  recently and has not grown for ``stall_s`` since its stage started. Its observed rate = fill
  growth over the last ``rate_window_s``.

Online correction for the simulator (``corrections``): a colour whose every slot is stalled ->
its contact sand is not reachable now (sim ``offsets``: only sand uncovered later counts); a
gaining slot -> its observed rate replaces the fill curve (``Slot.rate``) - the fitted curve is
pessimistic (gain recall 51 % on L189), a fresh bucket that fills fast must count as filling.
Waiting is allowed only while ``gaining()``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from state import ConveyorBucket


@dataclass
class TrackedSlot:
    stages: Tuple[Optional[str], ...]     # colours top first (None: not read yet)
    stage: int = 0
    fill: Optional[int] = None            # highest fill read of the current stage
    born: float = 0.0                     # tapped / first counted
    stage_t: float = 0.0                  # the current stage started (or the slot was counted)
    grow_t: Optional[float] = None        # growth last observed
    seen_t: Optional[float] = None        # last read with a fill
    hist: List[Tuple[float, int]] = field(default_factory=list)   # (t, fill), current stage
    tapped: bool = False                  # added by a tap of the bot
    expect_k: Optional[int] = None        # ... the counter has not shown it yet (k < this)
    done: bool = False                    # read at 100 % on its last stage: about to leave

    @property
    def colour(self) -> Optional[str]:
        return self.stages[self.stage] if self.stage < len(self.stages) else None

    @property
    def later(self) -> Tuple[str, ...]:
        return tuple(c for c in self.stages[self.stage + 1:] if c)

    def describe(self) -> str:
        f = "?" if self.fill is None else f"{self.fill}%"
        tower = "/" + "/".join(self.later) if self.later else ""
        return f"{self.colour or '?'}{tower} {f}"


class BeltBelief:
    def __init__(self, dc: Any) -> None:
        self.dc = dc
        self.slots: List[TrackedSlot] = []
        self.k: Optional[int] = None

    # -- input ----------------------------------------------------------------------------

    def tapped(self, colour: str, stages: Sequence[str], now: float) -> None:
        """The bot released a bucket / tower (colour = its top stage)."""
        waiting = sum(1 for s in self.slots if s.expect_k is not None)
        self.slots.append(TrackedSlot((colour,) + tuple(stages), fill=0, born=now,
                                      stage_t=now, hist=[(now, 0)], tapped=True,
                                      expect_k=(self.k or 0) + waiting + 1))

    def update(self, conveyor: Sequence[ConveyorBucket], k: Optional[int], now: float) -> None:
        """One frame's belt readings and the k/5 counter (None: not read)."""
        dc = self.dc
        matched: set = set()
        unmatched: List[ConveyorBucket] = []
        for b in sorted((b for b in conveyor if b.color and b.fill is not None),
                        key=lambda b: -b.fill):
            slot = self._match(b, matched)
            if slot is None:
                unmatched.append(b)
                continue
            matched.add(id(slot))
            self._read(slot, b, now)
        for b in unmatched:
            slot = self._advance_tower(b, matched)
            if slot is not None:
                matched.add(id(slot))
                self._read(slot, b, now)
        if k is not None:
            self.k = k
            for s in self.slots:   # the counter shows the tap (or never will: dead tap)
                if s.expect_k is not None and (k >= s.expect_k
                                               or now - s.born > dc.fresh_grace_s + dc.stall_s):
                    s.expect_k = None
            waiting = [s for s in self.slots if s.expect_k is not None]
            while len(self.slots) - len(waiting) > k:
                old = [s for s in self.slots if s.expect_k is None]
                self.slots.remove(max(old, key=lambda s: (
                    s.done, s.tapped and s.seen_t is None, id(s) not in matched,
                    -1 if s.fill is None else s.fill, -(s.seen_t or s.born))))
            spare = [b for b in unmatched if b.fill is not None]
            while len(self.slots) < k:
                b = spare.pop(0) if spare else None
                s = TrackedSlot(((b.color,) + tuple(b.stages)) if b is not None else (None,),
                                born=now, stage_t=now)
                if b is not None:
                    self._read(s, b, now)
                self.slots.append(s)
        for b in conveyor:   # an unread placeholder takes a colour read without a slot
            if b.color and not any(s.colour == b.color for s in self.slots):
                hole = next((s for s in self.slots if s.colour is None), None)
                if hole is not None:
                    hole.stages = (b.color,) + tuple(b.stages)
                    if b.fill is not None:
                        self._read(hole, b, now)

    def _match(self, b: ConveyorBucket, taken: set) -> Optional[TrackedSlot]:
        mine = [s for s in self.slots if s.colour == b.color and id(s) not in taken
                and not s.done]
        if not mine:
            return None
        below = [s for s in mine if (s.fill or 0) <= b.fill + 1]
        if below:
            return max(below, key=lambda s: (s.fill or 0, tuple(b.stages) == s.later))
        low = min(mine, key=lambda s: s.fill or 0)   # a reading a bit below: label jitter
        return low if (low.fill or 0) - b.fill <= self.dc.gain_min else None

    def _advance_tower(self, b: ConveyorBucket, taken: set) -> Optional[TrackedSlot]:
        """A reading of colour X with no slot of X: a tower whose next stage is X moved on."""
        for s in self.slots:
            if id(s) in taken or s.stage + 1 >= len(s.stages) or s.stages[s.stage + 1] != b.color:
                continue
            s.stage += 1
            s.fill, s.hist, s.stage_t, s.done = None, [], s.seen_t or s.stage_t, False
            s.grow_t = s.seen_t   # finishing a stage was progress
            return s
        return None

    def _read(self, s: TrackedSlot, b: ConveyorBucket, now: float) -> None:
        dc = self.dc
        f = b.fill
        if f >= 100:
            s.seen_t = s.grow_t = now
            if s.stage + 1 < len(s.stages):
                s.stage += 1
                s.fill, s.hist, s.stage_t = 0, [(now, 0)], now
            else:
                s.fill, s.done = 100, True
            return
        if s.fill is None or f >= s.fill + dc.gain_min:
            if s.fill is not None or s.tapped:
                s.grow_t = now
            s.fill = f if s.fill is None else max(s.fill, f)
        elif f > s.fill:
            s.fill = f            # +1 / +2: no growth yet, but the new reference
        s.seen_t = now
        s.hist.append((now, f))
        while len(s.hist) > 2 and now - s.hist[1][0] >= dc.rate_window_s:
            s.hist.pop(0)

    # -- queries --------------------------------------------------------------------------

    def gaining_slots(self, now: float) -> List[TrackedSlot]:
        dc = self.dc
        return [s for s in self.slots if not s.done and (
            (s.grow_t is not None and now - s.grow_t <= dc.gain_window_s)
            or (s.tapped and now - s.born <= dc.fresh_grace_s))]

    def gaining(self, now: float) -> bool:
        """A belt bucket is actually gaining sand: waiting can change something."""
        return bool(self.gaining_slots(now))

    def stalled(self, s: TrackedSlot, now: float) -> bool:
        dc = self.dc
        if s.done or s.seen_t is None or now - s.seen_t > dc.forget_s + dc.stall_s:
            return False
        since = max(s.stage_t, s.grow_t if s.grow_t is not None else s.stage_t)
        return now - since >= dc.stall_s

    def stalled_colours(self, now: float) -> List[str]:
        """Colours whose every unfinished slot is stalled."""
        out = []
        for c in {s.colour for s in self.slots if s.colour and not s.done}:
            mine = [s for s in self.slots if s.colour == c and not s.done]
            if all(self.stalled(s, now) for s in mine):
                out.append(c)
        return sorted(out)

    def rate(self, s: TrackedSlot) -> Optional[float]:
        """Observed fill %/s of the current stage over the rate window (None: too few reads)."""
        if len(s.hist) < 2:
            return None
        (t0, f0), (t1, f1) = s.hist[0], s.hist[-1]
        if t1 - t0 < 1.0:
            return None
        return max(0.0, (max(f for _, f in s.hist) - f0) / (t1 - t0))

    def buckets(self) -> List[ConveyorBucket]:
        """The belt as the simulator should start it: one entry per slot (done ones left out)."""
        return [ConveyorBucket(s.colour, s.fill, stages=s.later) for s in self.slots
                if not s.done]

    def corrections(self, now: float) -> Tuple[List[str], List[Optional[float]]]:
        """(stalled colours, per ``buckets()`` entry the observed rate of a gaining slot)."""
        gaining = {id(s) for s in self.gaining_slots(now)}
        rates = []
        for s in self.slots:
            if s.done:
                continue
            r = self.rate(s) if id(s) in gaining else None
            rates.append(r if r is not None and r > 0 else None)
        return self.stalled_colours(now), rates

    def describe(self, now: float) -> str:
        gain = {id(s) for s in self.gaining_slots(now)}
        parts = []
        for s in self.slots:
            tag = (" gaining" if id(s) in gain else " stalled" if self.stalled(s, now)
                   else "")
            r = self.rate(s)
            parts.append(s.describe() + tag + (f" {r:.0f}%/s" if r else ""))
        return ", ".join(parts) or "empty"
