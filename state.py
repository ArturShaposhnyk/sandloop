"""Level state model built from a vision.GameView (see docs/GAME_RULES.md).

Pure logic (no OpenCV): the grid of cells, which buckets are tappable, locks with their
counters, the conveyor (k of "k/5" and the buckets riding it), remaining and exposed sand per
colour, and the demand for sand per colour. ``after_tap`` simulates the immediate effect of a
tap for the strategy's lookahead.

Sand amounts are shares of the picture area (vision.read_sand). Bucket capacity is unknown and
differs between levels, but levels are balanced (sand_of_colour ~ buckets_of_colour x capacity),
so the capacity is estimated as remaining sand / pending buckets.
"""
from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from cells import Cell, Kind, exit_connected, path_playable

if TYPE_CHECKING:
    from vision import GameView

Pos = Tuple[int, int]
Grid = List[List[Optional[Cell]]]


@dataclass(frozen=True)
class Lock:
    """A lock in the grid: one ice block, lock icon or key, or one multi-cell region such as a
    curtain."""
    kind: str                     # "ice", "curtain", "lock", "key", ...
    unlock: str                   # "taps" (any bucket tapped) | "fills" (a bucket reaches 100%)
                                  # | "key" (the key of its colour arrives) | "path" (key: flies
                                  # when the path rule connects it to the exit)
    counter: Optional[int]        # number shown on it, None if not read
    cells: Tuple[Pos, ...]
    color: Optional[str] = None   # lock / key colour


@dataclass(frozen=True)
class ConveyorBucket:
    """One belt slot: a bucket, or a whole tower (one slot for all its stages; the top stage
    collects first, at 100 % the next stage starts collecting)."""
    color: Optional[str]          # the collecting bucket (a tower's current stage)
    fill: Optional[int]           # percent, None if no label was read
    uncertain: bool = False       # at the belt's edge without a label (vision.Slot.uncertain)
    stages: Tuple[str, ...] = ()  # tower: colours of the stages still to come, in order

    def need(self, unknown: float) -> float:
        """Share of a bucket's capacity still to fill (``unknown`` if the fill was not read)."""
        return unknown if self.fill is None else max(0.0, 1.0 - self.fill / 100.0)

    def need_for(self, color: str, unknown: float) -> float:
        """Capacities of ``color`` this slot still has to fill (current stage + later ones)."""
        return ((self.need(unknown) if self.color == color else 0.0)
                + sum(1.0 for s in self.stages if s == color))


@dataclass
class LevelState:
    grid: Grid
    exits: List[Pos]
    tappable: List[Pos]                       # path rule AND visual face-up agree
    disagreements: List[Pos]                  # visual state != path rule: never tapped
    locks: List[Lock]
    conveyor_k: Optional[int]                 # k of "k/5" (authoritative), None if not read
    conveyor_slots: int
    conveyor: List[ConveyorBucket]
    sand: Dict[str, float] = field(default_factory=dict)      # share of the picture
    exposed: Dict[str, float] = field(default_factory=dict)   # share of the contact layer
    unknown_fill_need: float = 0.5
    belt_stalled: Dict[str, float] = field(default_factory=dict)  # colour -> s without fill growth
    unlocked: List[Tuple[str, Tuple[Pos, ...]]] = field(default_factory=list)
    # ^ after_tap: (colour, lock cells) opened by keys that flew because of the simulated tap
    lower: Dict[str, float] = field(default_factory=dict)     # share of the picture's lower part

    # -- construction ---------------------------------------------------------------------

    @classmethod
    def from_view(cls, gv: "GameView", conveyor_slots: int = 5,
                  unknown_fill_need: float = 0.5,
                  belt_stalled: Optional[Dict[str, float]] = None) -> "LevelState":
        grid = copy.deepcopy(gv.grid)
        locks: List[Lock] = []
        for reg in gv.regions:
            locks.append(Lock(reg.lock, reg.unlock, reg.counter, tuple(sorted(reg.cells))))
        for row in grid:
            for cell in row:
                if cell is not None and cell.kind is Kind.LOCK and cell.region is None:
                    locks.append(Lock(cell.lock or "lock", cell.unlock or "", cell.counter,
                                      ((cell.row, cell.col),), cell.color))
        locks = _merge_lock_icons(locks)
        # the k/5 counter is authoritative: edge slots without a label may be end posts or
        # buckets entering/leaving, and at k=0 anything seen on the belt is decoration
        conveyor = [] if gv.counter == 0 else [
            ConveyorBucket(s.color, s.fill, s.uncertain, (s.body,) if s.body else ())
            for s in gv.slots if not s.uncertain]
        return cls(grid, gv.lattice.exits(), list(gv.tappable()), list(gv.disagreements), locks,
                   gv.counter, conveyor_slots, conveyor, dict(gv.sand), dict(gv.exposed),
                   unknown_fill_need, dict(belt_stalled or {}), lower=dict(gv.lower))

    # -- grid -----------------------------------------------------------------------------

    @property
    def rows(self) -> int:
        return len(self.grid)

    @property
    def cols(self) -> int:
        return len(self.grid[0]) if self.grid else 0

    def cell(self, pos: Pos) -> Optional[Cell]:
        r, c = pos
        return self.grid[r][c]

    def cells(self, kind: Optional[Kind] = None) -> List[Cell]:
        return [c for row in self.grid for c in row
                if c is not None and (kind is None or c.kind is kind)]

    def playable(self) -> List[Pos]:
        """Buckets connected to the exit by the path rule (ignores the visual state)."""
        play = path_playable(self.grid, self.exits)
        return [(r, c) for r in range(self.rows) for c in range(self.cols) if play[r][c]]

    def connected(self) -> Set[Pos]:
        """Occupied cells (buckets, locks, unknown) the path rule connects to the exit."""
        conn = exit_connected(self.grid, self.exits)
        return {(r, c) for r in range(self.rows) for c in range(self.cols) if conn[r][c]}

    # -- conveyor -------------------------------------------------------------------------

    @property
    def free_slots(self) -> Optional[int]:
        """Free conveyor slots from the "k/5" counter (None if it was not read)."""
        return None if self.conveyor_k is None else max(0, self.conveyor_slots - self.conveyor_k)

    def on_belt(self, color: str) -> bool:
        return any(b.color == color for b in self.conveyor)

    def conveyor_need(self, color: str) -> float:
        """Bucket capacities of ``color`` the conveyor still has to fill (tower stages too)."""
        return sum(b.need_for(color, self.unknown_fill_need) for b in self.conveyor)

    # -- sand vs. buckets -----------------------------------------------------------------

    def grid_buckets(self) -> Dict[str, int]:
        """Known buckets per colour still in the grid (a tower counts its top and body stage;
        buckets under curtains are hidden and counted by ``hidden_buckets``)."""
        n: Counter = Counter()
        for cell in self.cells(Kind.BUCKET):
            if cell.color:
                n[cell.color] += 1
            if cell.body:
                n[cell.body] += 1
        return dict(n)

    @property
    def hidden_buckets(self) -> int:
        """Buckets of unknown colour: one under every curtain cell and every lock cell."""
        return sum(len(lock.cells) for lock in self.locks if lock.kind in ("curtain", "lock"))

    def pending_buckets(self) -> float:
        """Bucket capacities still to fill: grid + hidden + what the conveyor still needs.
        Ice blocks are not counted (what is under the ice is not known yet)."""
        conv = sum(b.need(self.unknown_fill_need) + len(b.stages) for b in self.conveyor)
        return sum(self.grid_buckets().values()) + self.hidden_buckets + conv

    def capacity_estimate(self) -> Optional[float]:
        """Sand per bucket (share of the picture), assuming the level is balanced."""
        pending = self.pending_buckets()
        total = sum(self.sand.values())
        return total / pending if pending > 0 and total > 0 else None

    # -- simulation -----------------------------------------------------------------------

    def after_tap(self, pos: Pos) -> "LevelState":
        """The state right after tapping the bucket at ``pos`` (no sand movement): the bucket
        goes to the conveyor and its cell empties - a tower goes WHOLE, one belt slot for all
        its stages (GAME_RULES.md) - every lock that opens by taps counts down by one, and keys
        the emptied cell connects to the exit fly to their locks."""
        cell = self.cell(pos)
        if cell is None or cell.kind is not Kind.BUCKET:
            raise ValueError(f"no bucket at {pos}")
        nxt = copy.deepcopy(self)
        nxt._release_keys()   # keys already connected are flying anyway: not this tap's credit
        r, c = pos
        nxt.grid[r][c] = Cell(r, c, Kind.EMPTY)
        nxt.locks = [Lock(l.kind, l.unlock, max(0, l.counter - 1), l.cells)
                     if l.unlock == "taps" and l.counter is not None else l for l in self.locks]
        for lock in nxt.locks:
            for lr, lc in lock.cells:
                lcell = nxt.grid[lr][lc]
                if lcell is not None and lcell.kind is Kind.LOCK:
                    lcell.counter = lock.counter
        nxt.unlocked = nxt._release_keys()
        nxt.conveyor = self.conveyor + [
            ConveyorBucket(cell.color, 0, stages=(cell.body,) if cell.body else ())]
        nxt.conveyor_k = None if self.conveyor_k is None else self.conveyor_k + 1
        # predicted: a newly connected bucket flips face-up; cells that disagreed stay excluded
        nxt.disagreements = [p for p in self.disagreements if p != pos]
        nxt.tappable = [p for p in nxt.playable() if p not in nxt.disagreements]
        return nxt

    def _release_keys(self) -> List[Tuple[str, Tuple[Pos, ...]]]:
        """Keys the path rule connects to the exit fly to the lock of their colour: the key's
        cell empties (confirmed, GAME_RULES.md) and the lock's cells open (buckets of unknown
        colour until seen: UNKNOWN here). Repeats while a released key connects another key.
        A key whose colour has no lock, or several separate locks, opens nothing here."""
        opened: List[Tuple[str, Tuple[Pos, ...]]] = []
        while True:
            conn = self.connected()
            keys = [l for l in self.locks if l.kind == "key" and l.cells[0] in conn]
            if not keys:
                return opened
            for key in keys:
                (kr, kc), = key.cells
                self.grid[kr][kc] = Cell(kr, kc, Kind.EMPTY)
                self.locks.remove(key)
                targets = [l for l in self.locks if l.kind == "lock" and l.color == key.color]
                if len(targets) != 1:
                    continue
                target = targets[0]
                self.locks.remove(target)
                for lr, lc in target.cells:
                    self.grid[lr][lc] = Cell(lr, lc, Kind.UNKNOWN,
                                             reason="opened lock: bucket colour not seen yet")
                opened.append((key.color or "?", target.cells))

    def opened_by(self, pos: Pos) -> Tuple[List[Pos], List[Pos]]:
        """Lookahead of one tap: (buckets, lock cells) that become connected to the exit."""
        before = self.connected()
        after = self.after_tap(pos)
        new = sorted(p for p in after.connected() - before if p != pos)
        buckets = [p for p in new if after.cell(p).kind is Kind.BUCKET]
        locks = [p for p in new if after.cell(p).kind is Kind.LOCK]
        return buckets, locks

    def summary(self) -> List[str]:
        k = "?" if self.conveyor_k is None else self.conveyor_k
        cap = self.capacity_estimate()
        conv = ", ".join(f"{b.color}{'' if b.fill is None else f' {b.fill}%'}"
                         for b in self.conveyor) or "-"
        return [
            f"conveyor {k}/{self.conveyor_slots}: {conv}",
            "grid buckets: " + (", ".join(f"{c} {n}" for c, n in
                                          sorted(self.grid_buckets().items())) or "-")
            + (f", hidden {self.hidden_buckets}" if self.hidden_buckets else ""),
            "locks: " + (", ".join(f"{l.kind}{'' if l.counter is None else l.counter}"
                                   f"{'' if l.color is None else ':' + l.color}"
                                   f"@{'+'.join(f'r{r}c{c}' for r, c in l.cells[:1])}"
                                   for l in self.locks) or "-"),
            f"capacity estimate: {'-' if cap is None else f'{cap:.3f}'} of the picture / bucket",
        ]


def _merge_lock_icons(locks: List[Lock]) -> List[Lock]:
    """4-adjacent lock cells of the same colour form one lock (L183 has only single-cell ones;
    whether a lock can cover several cells is not confirmed yet)."""
    out = [l for l in locks if l.kind != "lock"]
    pending = [l for l in locks if l.kind == "lock"]
    while pending:
        group = [pending.pop(0)]
        grew = True
        while grew:
            grew = False
            for l in list(pending):
                if l.color == group[0].color and any(
                        abs(r - gr) + abs(c - gc) == 1
                        for (r, c) in l.cells for g in group for (gr, gc) in g.cells):
                    group.append(l)
                    pending.remove(l)
                    grew = True
        first = group[0]
        cells = tuple(sorted(p for g in group for p in g.cells))
        out.append(Lock(first.kind, first.unlock, first.counter, cells, first.color))
    return out


class BeltTracker:
    """Remembers the belt across frames: for every colour on the belt, how long its buckets'
    fill has not grown (a starving bucket = that colour's sand is not reachable right now).

    Per colour the key is (number of buckets with a read fill < 100%, sum of their fills). The
    clock restarts when that number changes (a bucket left at 100% or a new one arrived) or the
    sum grows by ``min_growth`` points since the last restart: the % labels jitter by one (99,
    98, 99), which is no growth. A colour that is not read in a frame (passing an edge, label
    unread) keeps its clock and stays reported for ``forget_s``; one seen with only finished
    buckets is dropped.
    """

    def __init__(self, min_growth: int = 2, forget_s: float = 3.0) -> None:
        self.min_growth, self.forget_s = min_growth, forget_s
        self._key: Dict[str, Tuple[int, int]] = {}
        self._since: Dict[str, float] = {}
        self._seen: Dict[str, float] = {}

    def update(self, conveyor: List[ConveyorBucket], now: float) -> Dict[str, float]:
        """Feed one frame's belt; returns colour -> seconds without fill growth (colours on the
        belt now, or out of sight for less than ``forget_s``)."""
        keys: Dict[str, Tuple[int, int]] = {}
        done = set()
        for b in conveyor:
            if b.color is None or b.fill is None:
                continue
            if b.fill >= 100:
                done.add(b.color)
                continue
            n, total = keys.get(b.color, (0, 0))
            keys[b.color] = (n + 1, total + b.fill)
        for color, (n, total) in keys.items():
            old = self._key.get(color)
            if old is None or old[0] != n or total >= old[1] + self.min_growth:
                self._key[color] = (n, total)
                self._since[color] = now
            self._seen[color] = now
        for color in list(self._key):
            if color not in keys and (color in done or now - self._seen[color] > self.forget_s):
                del self._key[color], self._since[color], self._seen[color]
        return {c: now - self._since[c] for c in self._key}
