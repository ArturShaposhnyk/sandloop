"""Unlock value and path planning toward useful buckets (docs/GAME_RULES.md, "Strategy").

Pure logic on ``state.LevelState``, used by ``strategy.choose`` (greedy + 1 with a path bonus):

* **Usefulness** of a colour: how much the belt could use one more bucket of it now or soon =
  max(contact-layer share, ``lower_weight`` x share in the picture's lower part) - what the
  same-colour belt buckets still claim (``belt_demand_exposure`` x need each); 0 if not enough
  sand of it remains. ``useful_min`` and up = useful.
* **Unlock value** of a tap: the buckets it connects to the exit, each with its colour and
  usefulness ("opens r3c3 cream, r4c2 darkred (useful)").
* **Path planning**: for every useful bucket the path rule does not connect yet, and every key
  whose colour has exactly one lock, the cheapest set of buckets to remove that connects it to
  the exit (Dijkstra over the grid; empty cells cost 0, a bucket costs 1 + more when its colour
  is barely exposed, more when its colour is stalled on the belt, a bucket that is not safe to
  release now - and every lock / unknown cell - is impassable). The path is then replayed tap by
  tap (``LevelState.after_tap``): every step must pass the release rules in the state it would
  be tapped in (free slot, the last-slot threshold, exposure with the growing belt demand,
  enough sand, no short tower stage), so a path whose buckets would fill the belt with starving
  buckets is never taken. Paths of one step are the tap's own unlock value and are not planned.
"""
from __future__ import annotations

import copy
import heapq
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from cells import Cell, Kind
from config import StrategyConfig
from state import LevelState, Pos

# why releasing the bucket at pos in this state is bad (None = safe); tappability not included
Check = Callable[[LevelState, Pos], Optional[str]]


def usefulness(state: LevelState, colour: Optional[str], sc: StrategyConfig) -> float:
    if not colour:
        return 0.0
    need = state.conveyor_need(colour)
    cap = state.capacity_estimate()
    if cap and state.sand.get(colour, 0.0) < (need + 1) * cap * sc.sufficiency_margin:
        return 0.0
    soon = max(state.exposed.get(colour, 0.0), sc.lower_weight * state.lower.get(colour, 0.0))
    return max(0.0, soon - need * sc.belt_demand_exposure)


@dataclass
class Opened:
    """A bucket a tap connects to the exit."""
    pos: Pos
    color: Optional[str]
    value: float                  # usefulness of its colour (after the tap)
    useful: bool

    def describe(self) -> str:
        r, c = self.pos
        return f"r{r}c{c} {self.color or '?'}" + (" (useful)" if self.useful else "")


def opened_value(after: LevelState, opens: List[Pos], sc: StrategyConfig) -> List[Opened]:
    """Usefulness of the buckets ``opens`` (connected by a tap) in the state ``after`` it."""
    out = []
    for p in opens:
        colour = after.cell(p).color
        u = usefulness(after, colour, sc)
        out.append(Opened(p, colour, u, u >= sc.useful_min))
    return out


@dataclass
class Plan:
    target: Pos
    label: str                    # "r5c1 darkred" | "key yellow"
    value: float                  # target: its colour's usefulness | path_key_value x lock cells
    steps: List[Pos]              # buckets to tap, in order from the exit
    cost: float                   # sum of the steps' release costs
    rank: float = 0.0             # value - path_step_cost x cost (the path bonus scales with it)
    rejected: Optional[str] = None

    def describe(self, step: int = 1, detail: bool = True) -> str:
        s = f"path to {self.label}: step {step}/{len(self.steps)}"
        if not detail:
            return s
        if self.rejected:
            return f"{s} - no: {self.rejected}"
        return s + f" (value {self.value:.2f}, cost {self.cost:.1f})"


def targets(state: LevelState, sc: StrategyConfig) -> List[Tuple[Pos, str, float]]:
    """(cell, label, value) of what is worth opening a path to."""
    conn = state.connected()
    out = []
    for cell in state.cells(Kind.BUCKET):
        pos = (cell.row, cell.col)
        if pos in conn:
            continue
        u = usefulness(state, cell.color, sc)
        if u >= sc.useful_min:
            out.append((pos, f"r{cell.row}c{cell.col} {cell.color}", u))
    for key in state.locks:
        if key.kind != "key" or key.cells[0] in conn:
            continue
        locks = [l for l in state.locks if l.kind == "lock" and l.color == key.color]
        if len(locks) == 1:   # several locks of its colour: which one opens is not known
            out.append((key.cells[0], f"key {key.color}", sc.path_key_value * len(locks[0].cells)))
    return out


def release_costs(state: LevelState, check: Check, sc: StrategyConfig) -> Dict[Pos, float]:
    """Cost of every cell a path may cross: empty 0, a bucket safe to release now 1 + up to
    ``path_cost_low_exposure`` the less its colour is exposed + ``path_cost_stalled`` if its
    colour is stalled on the belt (a lost slot: another path is preferred). Missing =
    impassable."""
    costs: Dict[Pos, float] = {}
    for r, row in enumerate(state.grid):
        for c, cell in enumerate(row):
            if cell is None:
                continue
            if cell.kind is Kind.EMPTY:
                costs[(r, c)] = 0.0
            elif cell.kind is Kind.BUCKET and check(state, (r, c)) is None:
                colour = cell.color or ""
                avail = (state.exposed.get(colour, 0.0)
                         - state.conveyor_need(colour) * sc.belt_demand_exposure)
                short = max(0.0, 1.0 - avail / sc.path_good_avail) if sc.path_good_avail else 0.0
                stalled = state.belt_stalled.get(colour, 0.0) >= sc.stall_s
                costs[(r, c)] = (1.0 + sc.path_cost_low_exposure * short
                                 + (sc.path_cost_stalled if stalled else 0.0))
    return costs


def shortest_path(state: LevelState, target: Pos, costs: Dict[Pos, float]
                  ) -> Optional[Tuple[float, List[Pos]]]:
    """Cheapest path from an exit cell to ``target`` (Dijkstra, cost of a cell paid on entering):
    (cost, buckets on it in order from the exit, target excluded), None if there is none."""
    def cost(p: Pos) -> Optional[float]:
        return 0.0 if p == target else costs.get(p)

    dist: Dict[Pos, float] = {}
    prev: Dict[Pos, Optional[Pos]] = {}
    heap: List[Tuple[float, Pos]] = []
    for e in map(tuple, state.exits):
        c = cost(e)
        if c is not None and c < dist.get(e, float("inf")):
            dist[e], prev[e] = c, None
            heapq.heappush(heap, (c, e))
    while heap:
        d, p = heapq.heappop(heap)
        if d > dist[p]:
            continue
        if p == target:
            path: List[Pos] = []
            q: Optional[Pos] = p
            while q is not None:
                path.append(q)
                q = prev[q]
            steps = [q for q in reversed(path)
                     if q != target and state.cell(q).kind is Kind.BUCKET]
            return d, steps
        r, c = p
        for n in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not (0 <= n[0] < state.rows and 0 <= n[1] < state.cols):
                continue
            nc = cost(n)
            if nc is not None and d + nc < dist.get(n, float("inf")):
                dist[n], prev[n] = d + nc, p
                heapq.heappush(heap, (d + nc, n))
    return None


def replay(state: LevelState, steps: List[Pos], check: Check) -> Optional[str]:
    """Tap the steps one by one: why the path is not safe, or None if every step passes the
    release rules in the state it would be tapped in. A step that opens a chest ends the plan
    unless it is the last one: the chest scatters buckets at random, the rest is unknown."""
    st = state
    for i, p in enumerate(steps, 1):
        r, c = p
        if p not in st.tappable:
            return f"step {i} r{r}c{c} not tappable"
        why = check(st, p)
        if why:
            return f"step {i} r{r}c{c}: {why}"
        st = st.after_tap(p)
        if st.chests_opened and i < len(steps):
            (cr, cc), n = st.chests_opened[0]
            return f"step {i} r{r}c{c} opens chest r{cr}c{cc} ({n} buckets scatter)"
    return None


@dataclass
class ScatterRisk:
    """What a chest opening may do (GAME_RULES.md, L186): its N buckets land in N of the free
    cells at random (worst case: they block paths)."""
    landing: int                  # free cells they can land in (empty cells + the chest cells)
    important: int                # ... of which blocking one disconnects an occupied cell
    buckets: int                  # buckets scattered (sum of N)
    cells: int                    # grid cells

    @property
    def expected_cuts(self) -> float:
        """Expected number of important cells hit."""
        if not self.landing:
            return 0.0
        return min(self.buckets, self.landing) * self.important / self.landing

    @property
    def free_share(self) -> float:
        return self.landing / self.cells if self.cells else 0.0


def scatter_risk(after: LevelState) -> Optional[ScatterRisk]:
    """Risk of the chests opened in ``after`` (a LevelState.after_tap result), None if none.
    Important free cell = blocking it alone disconnects an occupied cell (bucket, key, lock,
    chest) the path rule connects now."""
    if not after.chests_opened:
        return None
    chests = {p for p, _ in after.chests_opened}
    free = [(r, c) for r, row in enumerate(after.grid) for c, cell in enumerate(row)
            if cell is not None and (cell.kind is Kind.EMPTY or (r, c) in chests)]
    trial = copy.deepcopy(after)
    for p in chests:   # the chest cells are free again in the real game (landing spots)
        trial.grid[p[0]][p[1]] = Cell(p[0], p[1], Kind.EMPTY)
    base = trial.connected()
    important = 0
    for r, c in free:
        saved = trial.grid[r][c]
        trial.grid[r][c] = Cell(r, c, Kind.UNKNOWN)
        if not base - {(r, c)} <= trial.connected():
            important += 1
        trial.grid[r][c] = saved
    cells = sum(cell is not None for row in after.grid for cell in row)
    return ScatterRisk(len(free), important, sum(n for _, n in after.chests_opened), cells)


def plan(state: LevelState, check: Check, sc: StrategyConfig) -> List[Plan]:
    """Plans of 2+ steps to every target, best first; rejected ones (unsafe step, not worth
    its cost) last with the reason."""
    costs = release_costs(state, check, sc)
    plans: List[Plan] = []
    for pos, label, value in targets(state, sc):
        found = shortest_path(state, pos, costs)
        if found is None or len(found[1]) < 2:
            continue
        cost, steps = found
        p = Plan(pos, label, value, steps, cost, value - sc.path_step_cost * cost)
        p.rejected = replay(state, p.steps, check)
        if p.rejected is None and p.rank <= 0:
            p.rejected = f"not worth it (value {p.value:.2f} < cost)"
        plans.append(p)
    return sorted(plans, key=lambda p: (p.rejected is not None, -p.rank, len(p.steps)))
