"""Sampled worlds for the buckets a board hides (redesign phase C; decider.py, tools/selfplay.py).

Buckets of unknown colour - under ice / key locks / curtains, mystery buckets, the N buckets of
a tunnel (L190) or a chest (L186) - no longer end a plan ("horizon"): every sampled world gives
them colours drawn from the **colour balance** (levels are balanced, GAME_RULES.md "Sand per
bucket"): with one capacity for the whole level, colour c misses ~ its sand / capacity - its
known bucket stages; the hidden buckets are a bag of colours in those proportions (largest
remainder, the remainders drawn at random), shuffled per world. Colours an earlier attempt saw
under a lock (levelmemory) are fixed, not sampled.

``SampledSim`` plays such a world: a cell that opens (ice, key lock, curtain, mystery) shows its
sampled colour at once, a tunnel drops its sampled queue in order, an opened chest scatters its
sampled buckets over free cells (a per-world random order of the grid's cells).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Sequence, Tuple

from cells import Cell, Kind
from sim import LOCKED, UNREAD, Picture, Sim, SimConfig
from state import LevelState, Pos

Hidden = Dict[Pos, Tuple[str, Optional[str]]]


@dataclass
class Unknowns:
    """Where a board hides buckets: ``cells`` one bucket each (ice / key lock / curtain cells,
    mystery buckets), ``tunnels`` / ``chests`` cell -> buckets still inside."""
    cells: List[Pos] = field(default_factory=list)
    tunnels: Dict[Pos, int] = field(default_factory=dict)
    chests: Dict[Pos, int] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.cells) + sum(self.tunnels.values()) + sum(self.chests.values())


def find_unknowns(grid: LevelState, known: Sequence[Pos] = ()) -> Unknowns:
    out = Unknowns()
    for lock in grid.locks:
        if lock.kind in ("ice", "lock", "curtain", "mystery"):
            out.cells += [p for p in lock.cells if p not in known]
        elif lock.kind == "tunnel":
            out.tunnels[lock.cells[0]] = 1 if lock.counter is None else lock.counter
        elif lock.kind == "chest":
            out.chests[lock.cells[0]] = lock.counter or 1
    out.cells.sort()
    return out


def known_stages(s: Sim) -> Dict[str, float]:
    """Bucket stages per colour the board shows: grid buckets (towers: both stages), what the
    belt still needs, colours fixed under locks (memory)."""
    out: Dict[str, float] = {}
    for c in s.grid.cells(Kind.BUCKET):
        for col in (c.color, c.body):
            if col:
                out[col] = out.get(col, 0.0) + 1
    for cols in s.hidden.values():
        for col in cols:
            if col:
                out[col] = out.get(col, 0.0) + 1
    for slot in s.belt:
        for i in range(slot.stage, len(slot.stages)):
            col = slot.stages[i]
            if col == UNREAD:
                continue
            part = slot.left / max(1, s.capacity.get(col, 1)) if i == slot.stage else 1.0
            out[col] = out.get(col, 0.0) + part
    return out


def missing_colours(pic: Picture, known: Dict[str, float], hidden: int) -> Dict[str, float]:
    """Per colour the buckets the board does not show: its sand / the level-wide capacity -
    its known stages (>= 0). Colours whose sand fits no known bucket share the rest."""
    sand = {c: pic.count(c) for c in pic.names if c != LOCKED and pic.count(c) > 0}
    if not sand or hidden <= 0:
        return {}
    cap = sum(sand.values()) / max(1e-9, sum(known.get(c, 0.0) for c in sand) + hidden)
    miss = {c: max(0.0, n / cap - known.get(c, 0.0)) for c, n in sand.items()}
    if sum(miss.values()) <= 0:
        return {c: 1.0 for c in sand}
    return {c: v for c, v in miss.items() if v > 0}


def sample_bag(weights: Dict[str, float], n: int, rng: random.Random) -> List[str]:
    """``n`` colours in proportion to ``weights``: the whole parts, then the remainders drawn
    at random without replacement; shuffled."""
    if n <= 0 or not weights:
        return []
    total = sum(weights.values())
    exp = {c: n * w / total for c, w in sorted(weights.items())}
    bag = [c for c, e in exp.items() for _ in range(int(e))]
    rest = {c: e - int(e) for c, e in exp.items() if e - int(e) > 1e-9}
    while len(bag) < n and rest:
        pick = rng.random() * sum(rest.values())
        for c, w in rest.items():
            pick -= w
            if pick <= 0:
                break
        bag.append(c)
        del rest[c]
    while len(bag) < n:   # rounding: the heaviest colour
        bag.append(max(exp, key=exp.get))
    rng.shuffle(bag)
    return bag


@dataclass
class SampledSim(Sim):
    """A Sim whose hidden buckets have colours (one sampled world). Shared by clones."""
    tunnel_q: Dict[Pos, Tuple[int, List[str]]] = field(default_factory=dict)
    # ^ tunnel cell -> (buckets inside at the start, their colours in drop order)
    chest_bag: Dict[Pos, List[str]] = field(default_factory=dict)
    land_order: List[Pos] = field(default_factory=list)   # chest buckets land in this order

    def place(self, pos: Pos) -> None:
        before = {l.cells[0]: l.counter for l in self.grid.locks if l.kind == "tunnel"}
        super().place(pos)
        g, changed = self.grid, False
        for lock in g.locks:
            q = self.tunnel_q.get(lock.cells[0])
            old = before.get(lock.cells[0])
            if q is None or old is None or lock.counter is None or lock.counter >= old:
                continue
            i = q[0] - lock.counter - 1
            front = lock.front()
            if 0 <= i < len(q[1]) and front is not None:
                r, c = front
                g.grid[r][c] = Cell(r, c, Kind.BUCKET, color=q[1][i])
                changed = True
        for cpos, cols in self.chest_bag.items():
            cell = g.cell(cpos)
            if cell is None or cell.kind is not Kind.UNKNOWN:
                continue
            g.grid[cpos[0]][cpos[1]] = Cell(cpos[0], cpos[1], Kind.EMPTY)
            free = [p for p in self.land_order if g.cell(p) is not None
                    and g.cell(p).kind is Kind.EMPTY]
            for (r, c), col in zip(free, cols):
                g.grid[r][c] = Cell(r, c, Kind.BUCKET, color=col)
            changed = True
        if changed:
            g.tappable = [p for p in g.playable() if p not in g.disagreements]

    def _unknown(self, cells, why: str) -> None:
        super()._unknown(cells, why)   # curtains open between taps: show their colours too
        self._apply_hidden()


def worlds(state: LevelState, pic: Picture, sc: SimConfig, n: int, seed: int,
           capacity: Optional[Dict[str, int]] = None, memory: Optional[Hidden] = None,
           base: Optional[Sim] = None) -> List[SampledSim]:
    """``n`` sampled worlds of ``state`` + ``pic`` (one if the board hides nothing).
    ``capacity``: cells per stage measured in earlier attempts (levelmemory), kept.
    ``memory``: colours seen under locks before (fixed). ``base``: the root Sim to sample from
    (decider.py builds it from the belief), else ``Sim.start``."""
    root = base if base is not None else Sim.start(state, pic, sc, capacity, memory)
    unk = find_unknowns(root.grid, list(root.hidden))
    if unk.count == 0:
        return [_sampled(root, root.hidden, root.capacity, {}, {}, [])]
    known = known_stages(root)
    weights = missing_colours(root.pic, known, unk.count)
    cells = [(c.row, c.col) for row in root.grid.grid for c in row if c is not None]
    out = []
    for i in range(n):
        rng = random.Random(seed * 7919 + i)
        bag = sample_bag(weights, unk.count, rng)
        if not bag:
            return [_sampled(root, root.hidden, root.capacity, {}, {}, [])]
        hidden = dict(root.hidden)
        hidden.update({p: (bag.pop(), None) for p in unk.cells})
        tq = {p: (k, [bag.pop() for _ in range(k)]) for p, k in sorted(unk.tunnels.items())}
        chests = {p: [bag.pop() for _ in range(k)] for p, k in sorted(unk.chests.items())}
        order = list(cells)
        rng.shuffle(order)
        stages = dict(known)
        for col in ([v[0] for p, v in hidden.items() if p not in root.hidden]
                    + [c for _, q in tq.values() for c in q] + [c for q in chests.values() for c in q]):
            stages[col] = stages.get(col, 0.0) + 1
        cap = dict(root.capacity)
        cap.update({c: max(1, round(root.pic.count(c) / k)) for c, k in stages.items()
                    if k > 0 and root.pic.count(c) > 0})
        cap.update(capacity or {})
        out.append(_sampled(root, hidden, cap, tq, chests, order))
    return out


def _sampled(root: Sim, hidden: Hidden, cap: Dict[str, int], tq, chests, order) -> SampledSim:
    s = SampledSim(**{f.name: getattr(root, f.name) for f in fields(Sim)},
                   tunnel_q=tq, chest_bag=chests, land_order=order)
    s.hidden, s.capacity = hidden, cap
    s = s.clone()   # own belt slots and picture
    if s.belt:      # the belt's cells left follow this world's capacity
        for slot in s.belt:
            old = root.capacity.get(slot.colour, 1)
            new = cap.get(slot.colour, old)
            if new != old and slot.colour != UNREAD:
                slot.left = max(1, round(slot.left * new / max(1, old)))
    return s
