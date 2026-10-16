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
    """A lock in the grid: one ice block, or one multi-cell region such as a curtain."""
    kind: str                     # "ice", "curtain", ...
    unlock: str                   # "taps" (any bucket tapped) | "fills" (a bucket reaches 100%)
    counter: Optional[int]        # number shown on it, None if not read
    cells: Tuple[Pos, ...]


@dataclass(frozen=True)
class ConveyorBucket:
    color: Optional[str]
    fill: Optional[int]           # percent, None if no label was read
    uncertain: bool = False       # at the belt's edge without a label (vision.Slot.uncertain)

    def need(self, unknown: float) -> float:
        """Share of a bucket's capacity still to fill (``unknown`` if the fill was not read)."""
        return unknown if self.fill is None else max(0.0, 1.0 - self.fill / 100.0)


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
    exposed: Dict[str, float] = field(default_factory=dict)   # share of the bottom band
    unknown_fill_need: float = 0.5

    # -- construction ---------------------------------------------------------------------

    @classmethod
    def from_view(cls, gv: "GameView", conveyor_slots: int = 5,
                  unknown_fill_need: float = 0.5) -> "LevelState":
        grid = copy.deepcopy(gv.grid)
        locks: List[Lock] = []
        for reg in gv.regions:
            locks.append(Lock(reg.lock, reg.unlock, reg.counter, tuple(sorted(reg.cells))))
        for row in grid:
            for cell in row:
                if cell is not None and cell.kind is Kind.LOCK and cell.region is None:
                    locks.append(Lock(cell.lock or "lock", cell.unlock or "", cell.counter,
                                      ((cell.row, cell.col),)))
        # the k/5 counter is authoritative: edge slots without a label may be end posts or
        # buckets entering/leaving, and at k=0 anything seen on the belt is decoration
        conveyor = [] if gv.counter == 0 else [ConveyorBucket(s.color, s.fill, s.uncertain)
                                               for s in gv.slots if not s.uncertain]
        return cls(grid, gv.lattice.exits(), list(gv.tappable()), list(gv.disagreements), locks,
                   gv.counter, conveyor_slots, conveyor, dict(gv.sand), dict(gv.exposed),
                   unknown_fill_need)

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

    def conveyor_need(self, color: str) -> float:
        """Bucket capacities of ``color`` the conveyor still has to fill."""
        return sum(b.need(self.unknown_fill_need) for b in self.conveyor if b.color == color)

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
        """Buckets of unknown colour: one under every curtain cell."""
        return sum(len(lock.cells) for lock in self.locks if lock.kind == "curtain")

    def pending_buckets(self) -> float:
        """Bucket capacities still to fill: grid + hidden + what the conveyor still needs.
        Ice blocks are not counted (what is under the ice is not known yet)."""
        conv = sum(b.need(self.unknown_fill_need) for b in self.conveyor)
        return sum(self.grid_buckets().values()) + self.hidden_buckets + conv

    def capacity_estimate(self) -> Optional[float]:
        """Sand per bucket (share of the picture), assuming the level is balanced."""
        pending = self.pending_buckets()
        total = sum(self.sand.values())
        return total / pending if pending > 0 and total > 0 else None

    # -- simulation -----------------------------------------------------------------------

    def after_tap(self, pos: Pos) -> "LevelState":
        """The state right after tapping the bucket at ``pos`` (no sand movement): the bucket
        goes to the conveyor, its cell empties (a tower keeps its next stage, locked until the
        top bucket is 100%), and every lock that opens by taps counts down by one."""
        cell = self.cell(pos)
        if cell is None or cell.kind is not Kind.BUCKET:
            raise ValueError(f"no bucket at {pos}")
        nxt = copy.deepcopy(self)
        r, c = pos
        if cell.body:
            nxt.grid[r][c] = Cell(r, c, Kind.LOCK, color=cell.body, lock="tower",
                                  unlock="fills", counter=1)
        else:
            nxt.grid[r][c] = Cell(r, c, Kind.EMPTY)
        nxt.locks = [Lock(l.kind, l.unlock, max(0, l.counter - 1), l.cells)
                     if l.unlock == "taps" and l.counter is not None else l for l in self.locks]
        for lock in nxt.locks:
            for lr, lc in lock.cells:
                lcell = nxt.grid[lr][lc]
                if lcell is not None and lcell.kind is Kind.LOCK:
                    lcell.counter = lock.counter
        nxt.conveyor = self.conveyor + [ConveyorBucket(cell.color, 0)]
        nxt.conveyor_k = None if self.conveyor_k is None else self.conveyor_k + 1
        # predicted: a newly connected bucket flips face-up; cells that disagreed stay excluded
        nxt.disagreements = [p for p in self.disagreements if p != pos]
        nxt.tappable = [p for p in nxt.playable() if p not in nxt.disagreements]
        return nxt

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
                                   f"@{'+'.join(f'r{r}c{c}' for r, c in l.cells[:1])}"
                                   for l in self.locks) or "-"),
            f"capacity estimate: {'-' if cap is None else f'{cap:.3f}'} of the picture / bucket",
        ]
