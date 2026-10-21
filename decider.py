"""One decision maker (redesign phase C, docs/STATUS.md; ``decider: search`` in config.yaml).

Every move is picked by a receding-horizon search in the fitted simulator (sim.py), started
from the **belief** (belief.py: belt slots by identity, observed fill rates, stalls) and run
over K **sampled worlds** for the buckets the board hides (worlds.py). Only game rules are hard:
the path rule, a free belt slot, locks / ice / curtains / tunnels / chests / walls, shells and
towers - all exact in the simulator. Perception guards stay hard: the first tap must be in the
frame's ``state.tappable`` (face and path rule agree, colour confident after an unlock, not
under a hint arrow, not a dead tap, not flying) and the counter must be read.

The old vetoes are **score terms** (``decider:`` weights), averaged over the worlds:
25 % / exposure rules and "don't tap a shell bucket early" -> belt slot-seconds collecting
nothing (``weight_idle``) and slots stuck at the end (``weight_stuck``); sand margins, "tower
top must fill", gamble margins -> belt buckets or later tower stages short of sand
(``weight_dead``); slot reserves / key colours / ice reserve -> exposed colours with a bucket
waiting and no free slot (``weight_deficit``); a jam or a belt full of dead buckets
(``weight_lost``); progress: sand collected, buckets released, cells opened.

**Never idle**: waiting is a candidate only while a belt bucket is really gaining sand
(belief); otherwise the best tap is made, whatever its score. With no legal tap the bot waits.

Search: beam over action sequences (tap a bucket / wait until a bucket leaves), each node holds
one Sim per world and scores their mean; every first action keeps its best node in the beam,
so all of them are compared at the same depth. The first action of the best node is played;
consecutive taps after it (no wait between) go in the same burst.
"""
from __future__ import annotations

import time
import zlib
from dataclasses import dataclass, field
from itertools import dropwhile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from belief import BeltBelief
from cells import Kind
from sim import UNREAD, Picture, Sim, SimConfig, grid_signature
from state import LevelState, Pos

Action = Tuple[str, Optional[Pos]]          # ("tap", pos) | ("wait", None)
WAIT: Action = ("wait", None)
Rank = Tuple[float, float, float, float]    # (won, -lost, -dead buckets, value), world means


def search_config(cfg: Any, slots: int, x2: bool) -> SimConfig:
    """The planner's fitted simulator; shell buckets may go any time (their idle is priced)."""
    from levelplan import sim_config

    sc = sim_config(cfg.planner, slots, x2)
    sc.strict_shells = False
    sc.max_wait_s = cfg.decider.wait_max_s
    return sc


# --- score -------------------------------------------------------------------------------------


def _rate(s: Sim, colour: str) -> float:
    share = s.pic.share(colour)
    if s.offsets and colour in s.offsets:
        share = max(0.0, share - s.offsets[colour])
    return s.sc.rate(share)


def stuck(s: Sim) -> int:
    """Belt buckets collecting nothing now (stall corrections included)."""
    return sum(1 for b in s.belt if b.colour != UNREAD and (
        (b.shell and s.pic.shell_down(b.colour) is None) or _rate(s, b.colour) <= 0))


def deficit(s: Sim) -> int:
    """Colours exposed enough to fill, not collected by the belt, with a tappable bucket -
    more than the free slots (what the old key / ice / endgame reserves kept slots for)."""
    on_belt = {b.colour for b in s.belt}
    want = set()
    for p in s.grid.tappable:
        c = s.grid.cell(p)
        if (c.kind is Kind.BUCKET and c.color and c.color not in on_belt
                and _rate(s, c.color) >= s.sc.idle_rate):
            want.add(c.color)
    return max(0, len(want) - max(0, s.free))


def lost(s: Sim) -> bool:
    return s.jam or (s.free <= 0 and bool(s.belt) and s.short_slots() >= len(s.belt))


def rank(sims: Sequence[Sim], total: int, dc: Any) -> Rank:
    """Lines compare as the level planner's did (a dead belt bucket never leaves: any line
    with fewer is better, user run 171133): a win first, then not lost, then fewer belt
    buckets short of sand, then the score - each a mean over the sampled worlds."""
    n = len(sims)
    return (sum(s.won for s in sims) / n, -sum(lost(s) for s in sims) / n,
            -sum(s.short_slots() for s in sims) / n, sum(value(s, total, dc) for s in sims) / n)


def value(s: Sim, total: int, dc: Any) -> float:
    return (dc.weight_collect * s.collected / max(1, total)
            + (dc.weight_win if s.won else 0.0) - (dc.weight_lost if lost(s) else 0.0)
            - dc.weight_dead * s.short_slots() - dc.weight_stuck * stuck(s)
            - dc.weight_idle * s.idle + dc.weight_free * max(0, s.free)
            + dc.weight_tap * len(s.taps) + dc.weight_open * s.opened
            - dc.weight_full * s.full_taps - dc.weight_deficit * deficit(s))


def fmt_rank(r: Rank) -> str:
    out = [f"{r[3]:.2f}"]
    if r[2]:
        out.insert(0, f"dead {-r[2]:.1f}")
    if r[1]:
        out.insert(0, f"lost {-r[1]:.0%}")
    if r[0]:
        out.insert(0, f"win {r[0]:.0%}")
    return " ".join(out)


# --- search ------------------------------------------------------------------------------------


@dataclass
class Node:
    sims: List[Sim]
    first: Action
    path: Tuple[Action, ...]
    rank: Rank
    done: bool = False


@dataclass
class SearchResult:
    values: Dict[Action, Rank]
    paths: Dict[Action, Tuple[Action, ...]]
    nodes: int = 0
    depth: int = 0
    worlds: int = 1
    seconds: float = 0.0
    ends: Dict[Action, Sim] = field(default_factory=dict)   # world 0 at the end of each line
    seeded: bool = False              # the last decision's line beat the beam's

    def ranked(self) -> List[Tuple[Action, Rank]]:
        return sorted(self.values.items(), key=lambda kv: (tuple(-x for x in kv[1]),
                                                           kv[0][0] != "wait",
                                                           kv[0][1] or (-1, -1)))


def _step(s: Sim, act: Action, idle_laps: int, colour: Optional[str] = None) -> Sim:
    """One action in one world. A tap that is not possible in this world (a sampled colour
    changed the board, no free slot) or whose bucket is not ``colour`` here (the line was
    found in another world: the bot would see the other colour) passes the tap's belt time."""
    if act[0] == "tap":
        if (s.free > 0 and act[1] in s.moves()
                and (colour is None or s.grid.cell(act[1]).color == colour)):
            return s.tap(act[1])
        nxt = s.clone()
        for _ in range(s.sc.laps_per_tap):
            nxt.lap()
        return nxt
    w = s.wait()
    if w is None:                     # nothing moves in this world: time passes, idle
        w = s.clone()
        for _ in range(idle_laps):
            w.lap()
    return w


def play_line(root: Sim, path: Sequence[Action], idle_laps: int,
              colours: Optional[Sequence[str]] = None) -> Sim:
    """``root`` after the actions of ``path`` (``colours``: the colour of each tap in the
    world the line was found in)."""
    s, i = root, 0
    for act in path:
        if s.jam or s.won:
            break
        c = None
        if act[0] == "tap" and colours is not None:
            c = colours[i] if i < len(colours) else None
            i += 1
        s = _step(s, act, idle_laps, c)
    return s


def search(roots: Sequence[Sim], first: Sequence[Action], dc: Any, budget_s: float,
           total: Optional[int] = None, seed: Sequence[Action] = ()) -> SearchResult:
    """``first``: the allowed first actions. Beam search in the first world ``roots[0]``
    (most of the budget); ``seed``: the rest of the last decision's best line, scored too
    (warm start: the beam need not find it again). Then every first action's best line is
    played in all the sampled worlds (a tap of a bucket whose colour differs there passes its
    time) and scored by the mean - a line that works for one guess of the hidden colours
    only ranks lower."""
    t0 = time.monotonic()
    total = total or max(1, roots[0].pic.total() + roots[0].collected)
    idle_laps = max(1, round(dc.wait_idle_s / roots[0].sc.lap_s))
    res = _beam(roots[:1], first, dc, budget_s * (0.7 if len(roots) > 1 else 1.0), total)
    seed = tuple(seed)
    if seed and seed[0] in first:
        end = play_line(roots[0], seed, idle_laps)
        r = rank([end], total, dc)
        if seed[0] not in res.values or r > res.values[seed[0]]:
            res.values[seed[0]], res.paths[seed[0]], res.ends[seed[0]] = r, seed, end
            res.seeded = True
    if len(roots) > 1:
        for a, path in res.paths.items():
            colours = [c for kind, _, c, _ in res.ends[a].actions if kind == "tap"]
            ends = [res.ends[a]] + [play_line(w, path, idle_laps, colours) for w in roots[1:]]
            res.nodes += len(path) * (len(roots) - 1)
            res.values[a] = rank(ends, total, dc)
        res.worlds = len(roots)
    res.seconds = time.monotonic() - t0
    return res


def _beam(roots: Sequence[Sim], first: Sequence[Action], dc: Any, budget_s: float,
          total: int) -> SearchResult:
    t0 = time.monotonic()
    idle_laps = max(1, round(dc.wait_idle_s / roots[0].sc.lap_s))

    def score(sims: List[Sim]) -> Rank:
        return rank(sims, total, dc)

    beam: List[Node] = []
    nodes = 0
    for a in first:
        sims = [_step(s, a, idle_laps) for s in roots]
        nodes += len(sims)
        beam.append(Node(sims, a, (a,), score(sims), _terminal(sims)))
    best = {n.first: n for n in beam}
    depth = 1
    while depth < dc.max_depth and time.monotonic() - t0 < budget_s:
        children: List[Node] = [n for n in beam if n.done]
        for n in beam:
            if n.done:
                continue
            acts = sorted({("tap", p) for s in n.sims if not (s.jam or s.won)
                           for p in s.moves()})
            if any(s.belt and not s.jam for s in n.sims):
                acts.append(WAIT)
            for a in acts:
                sims = [s if (s.jam or s.won) else _step(s, a, idle_laps) for s in n.sims]
                nodes += len(sims)
                children.append(Node(sims, n.first, n.path + (a,), score(sims),
                                     _terminal(sims)))
        if all(n.done for n in children):
            break
        beam = _select(children, dc.beam)
        depth += 1
        for n in beam:   # the deepest surviving node per first action
            best[n.first] = n
    out = SearchResult({a: n.rank for a, n in best.items()},
                       {a: n.path for a, n in best.items()}, nodes, depth, len(roots),
                       time.monotonic() - t0)
    out.ends = {a: n.sims[0] for a, n in best.items()}
    return out


def _terminal(sims: List[Sim]) -> bool:
    return all(s.won or s.jam for s in sims)


def _select(children: List[Node], width: int) -> List[Node]:
    """The best node of every first action, then the best others (one per board)."""
    ranked = sorted(children, key=lambda n: tuple(-x for x in n.rank))
    out, firsts, seen = [], set(), set()
    for n in ranked:
        if n.first not in firsts:
            firsts.add(n.first)
            out.append(n)
            seen.add(_key(n))
    extra = 0
    for n in ranked:
        if extra >= width:
            break
        k = _key(n)
        if k in seen:
            continue
        seen.add(k)
        out.append(n)
        extra += 1
    return out


def _key(n: Node) -> Tuple:
    s = n.sims[0]
    return (s.signature(), tuple((b.colour, b.stage) for b in s.belt))


# --- live ----------------------------------------------------------------------------------------


@dataclass
class _Last:
    key: Tuple
    t: float
    result: SearchResult
    choice: Optional[Action] = None


class SearchDecider:
    """Per level: the belief of the belt and the search (Analyzer, tools/selfplay.py)."""

    def __init__(self, cfg: Any, memory_path: Optional[Path] = None) -> None:
        self.cfg, self.dc = cfg, cfg.decider
        self.memory_path = memory_path
        self.reset()

    def reset(self) -> None:
        self.belief = BeltBelief(self.dc)
        self.signature: Optional[str] = None
        self.x2 = False
        self.last: Optional[_Last] = None
        self.line: Tuple[Action, ...] = ()       # the rest of the chosen line (warm start)
        self.searches = 0
        self.log: List[Dict[str, Any]] = []      # levels.jsonl "searches"
        self.memory: Dict[Pos, Tuple[str, Optional[str]]] = {}
        self.capacity_shares: Dict[str, float] = {}
        self.attempt = 0

    # -- inputs -------------------------------------------------------------------------------

    def start(self, signature: str) -> None:
        """The level's board signature (levelmemory): colours seen under locks, capacities and
        the attempt number (it varies the sampled worlds on Try Again)."""
        if self.signature is not None:
            return
        self.signature = signature
        if self.memory_path is not None and Path(self.memory_path).exists():
            from levelmemory import LevelMemory

            mem = LevelMemory(Path(self.memory_path))
            self.memory = mem.revealed(signature)
            self.capacity_shares = mem.capacity_shares(signature)
            self.attempt = len(mem.attempts(signature))

    def set_speed(self, x2: bool, log=print) -> None:
        if x2 != self.x2:
            self.x2, self.last = x2, None

    def tapped(self, pos: Pos, colour: str, stages: Sequence[str], now: float) -> None:
        self.belief.tapped(colour, stages, now)
        self.last = None
        line = list(self.line)
        while line and line[0] == WAIT:
            line.pop(0)
        self.line = tuple(line[1:]) if line and line[0] == ("tap", tuple(pos)) else ()

    def observe(self, state: LevelState, now: float) -> None:
        self.belief.update(state.conveyor, state.conveyor_k, now)

    # -- the decision ---------------------------------------------------------------------------

    def decide(self, state: LevelState, picture: Callable[[], Picture], now: float, log=print):
        from strategy import Decision

        dc = self.dc
        if state.conveyor_k is None:
            return Decision(None, "wait: conveyor counter not read")
        gaining = self.belief.gaining(now)
        free = state.free_slots or 0
        legal = [p for p in state.tappable if free > 0 and state.cell(p) is not None
                 and state.cell(p).kind is Kind.BUCKET and state.cell(p).color]
        if not legal:
            why = "conveyor full" if free <= 0 else "no tappable bucket"
            return Decision(None, f"wait: {why} (belt: {self.belief.describe(now)})")
        stalled, rates = self.belief.corrections(now)
        key = (grid_signature(state.grid), state.conveyor_k, gaining, tuple(stalled),
               tuple(legal), self.x2)
        if self.last is not None and self.last.key == key and now - self.last.t < dc.replan_s:
            res = self.last.result
        else:
            res = self._search(state, picture(), legal, gaining, stalled, rates, now)
            prev = self.last.choice if self.last is not None else None
            self.last = _Last(key, now, res, prev)
        return self._choose(state, res, gaining, now, log)

    def _search(self, state: LevelState, pic: Picture, legal: List[Pos], gaining: bool,
                stalled: List[str], rates: List[Optional[float]], now: float) -> SearchResult:
        from worlds import worlds

        dc = self.dc
        st = state.fork()
        st.conveyor = self.belief.buckets()
        sc = search_config(self.cfg, state.conveyor_slots, self.x2)
        cap = None
        sand = sum(state.sand.values())
        if self.capacity_shares and sand > 0:   # memory: picture share -> cells
            cap = {c: max(1, round(v * pic.total() / sand))
                   for c, v in self.capacity_shares.items()}
        base = Sim.start(st, pic, sc, cap, self.memory)
        base.grid.disagreements = []
        base.grid.tappable = base.grid.playable()
        slots = iter(base.belt)
        for b, r in zip(st.conveyor, rates):   # Sim.start keeps the coloured, unfinished ones
            if not b.color or (b.fill is not None and b.fill >= 100):
                continue
            slot = next(slots, None)
            if slot is not None and r is not None:
                slot.rate, slot.rate_until = r, dc.override_s
        base.offsets = {c: base.pic.share(c) for c in stalled}
        seed = (dc.seed * 1_000_003 + zlib.crc32((self.signature or "").encode())
                + 7 * self.attempt) & 0x7FFFFFFF
        roots = worlds(st, pic, sc, dc.particles, seed, cap, self.memory, base=base)
        moves = set(roots[0].moves()) | {p for r in roots for p in r.moves()}
        first: List[Action] = [("tap", p) for p in legal if p in moves]
        if gaining or not first:
            first.append(WAIT)
        budget = dc.start_budget_s if self.searches == 0 else dc.budget_s
        self.searches += 1
        seed = self.line
        if seed and seed[0] not in first:      # its wait is over (or not allowed): next tap
            seed = tuple(dropwhile(lambda a: a == WAIT, seed))
        res = search(roots, first, dc, budget, seed=seed)
        self.log.append({"t": round(now, 2), "k": state.conveyor_k, "gaining": gaining,
                         "stalled": stalled, "worlds": res.worlds, "nodes": res.nodes,
                         "depth": res.depth, "s": round(res.seconds, 2),
                         "values": [[a[0], list(a[1]) if a[1] else None,
                                     [round(x, 3) for x in v]] for a, v in res.ranked()[:6]]})
        return res

    def _choose(self, state: LevelState, res: SearchResult, gaining: bool, now: float, log):
        from strategy import Candidate, Decision

        ranked = res.ranked()
        if not ranked:
            return Decision(None, "wait: search found no action")
        allowed = [(a, v) for a, v in ranked if a != WAIT or gaining]
        taps = [(a, v) for a, v in ranked if a[0] == "tap"]
        best, v = allowed[0] if allowed else ranked[0]
        prev = self.last.choice if self.last is not None else None
        pv = res.values.get(prev) if prev is not None else None
        if (pv is not None and prev != best and (prev != WAIT or gaining)
                and pv[:3] == v[:3] and pv[3] >= v[3] - self.dc.keep_margin):
            best, v = prev, pv                         # stable for the decision gate
        if self.last is not None:
            self.last.choice = best
        self.line = res.paths.get(best, (best,))
        text = self._summary(res, ranked)
        cands = []
        for a, val in taps:
            cell = state.cell(a[1])
            cands.append(Candidate(a[1], cell.color or "?", score=val[3],
                                   exposed=state.exposed.get(cell.color or "", 0.0),
                                   stages=[cell.body] if cell.body else []))
        belt = self.belief.describe(now)
        if best == WAIT:
            return Decision(None, f"wait: search - the belt is gaining ({belt}); {text}", cands)
        move = next(c for c in cands if c.pos == best[1])
        path = res.paths.get(best, (best,))
        n = sum(1 for a in path if a[0] == "tap")
        move.level_plan = f"search 1/{n}"
        move.follow = self._burst(state, path, cands)
        why = "" if gaining else " - no belt bucket is gaining, waiting would idle"
        return Decision(move, f"tap r{best[1][0]}c{best[1][1]} {move.color} (search "
                              f"{fmt_rank(v)}{why}; {text})", cands)

    def _burst(self, state: LevelState, path: Tuple[Action, ...], cands) -> List[Any]:
        """Taps that follow the first one at once in the best line (no wait between)."""
        from strategy import Candidate

        out: List[Any] = []
        free = (state.free_slots or 0) - 1
        st = state
        for i, a in enumerate(path[1:], 2):
            if a[0] != "tap" or len(out) + 1 >= self.cfg.planner.burst_max or free <= 1:
                break   # a follow-up never takes the last free slot (bot.burst_min_free)
            st = st.after_tap(path[i - 2][1])
            if a[1] not in state.tappable or a[1] not in st.tappable:
                break
            cell = st.cell(a[1])
            c = Candidate(a[1], cell.color or "?", stages=[cell.body] if cell.body else [],
                          exposed=state.exposed.get(cell.color or "", 0.0))
            c.level_plan = f"search {i}/{sum(1 for x in path if x[0] == 'tap')}"
            out.append(c)
            free -= 1
        return out

    @staticmethod
    def _summary(res: SearchResult, ranked: List[Tuple[Action, float]]) -> str:
        top = ", ".join(("wait" if a == WAIT else f"r{a[1][0]}c{a[1][1]}") + f" {fmt_rank(v)}"
                        for a, v in ranked[:4])
        return (f"{top}; {res.worlds} world(s), depth {res.depth}, {res.nodes} sims, "
                f"{res.seconds:.1f}s{', last line kept' if res.seeded else ''}")
