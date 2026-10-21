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
from typing import Any, TYPE_CHECKING, Dict, List, Optional, Set, Tuple

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
                                  # | "key" (the key of its colour arrives) | "drops" (L190 tunnel) | "path" (key: flies
                                  # when the path rule connects it to the exit)
    counter: Optional[int]        # number shown on it, None if not read
    cells: Tuple[Pos, ...]
    color: Optional[str] = None   # lock / key colour
    opening: Optional[str] = None  # tunnel (L190): the side its buckets drop out of

    def front(self) -> Optional[Pos]:
        """Tunnel: the cell in front of its opening (where its next bucket drops)."""
        from cells import TUNNEL_OPENINGS

        d = TUNNEL_OPENINGS.get(self.opening or "")
        if d is None:
            return None
        (r, c), = self.cells
        return r + d[0], c + d[1]


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
    belt_tracked: int = 0              # unfinished belt buckets BeltTracker holds (read recently)
    unlocked: List[Tuple[str, Tuple[Pos, ...]]] = field(default_factory=list)
    gates_opened: List[Tuple[str, Tuple[Pos, ...]]] = field(default_factory=list)   # chain locks
    # ^ after_tap: (colour, lock cells) opened by keys that flew because of the simulated tap
    chests_opened: List[Tuple[Pos, int]] = field(default_factory=list)
    # ^ after_tap: chests the simulated tap connects (cell, number of buckets they scatter)
    revealed: List[Pos] = field(default_factory=list)   # after_tap: mystery buckets revealed
    lower: Dict[str, float] = field(default_factory=dict)     # share of the picture's lower part
    shells: List[Any] = field(default_factory=list)     # picture.PicShell in the picture (L188)
    pic_locks: List[Any] = field(default_factory=list)  # picture.PicLock still locked (L188)
    locked: float = 0.0                # share of the picture behind picture locks
    pics_opened: List[str] = field(default_factory=list)   # after_tap: picture locks opened
    above: Dict[str, Dict[str, float]] = field(default_factory=dict)
    colour_capacity: Dict[str, float] = field(default_factory=dict)
    # ^ picture share one bucket stage of a colour takes, measured at level start (Analyzer):
    #   the picture reading under-counts some colours (L187 darkred, cream) by 2-3x
    # ^ contact colour -> colour right above it -> contact share it becomes once collected

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
                                      ((cell.row, cell.col),), cell.color, cell.opening))
        locks = _merge_lock_icons(locks)
        # the k/5 counter is authoritative: edge slots without a label may be end posts or
        # buckets entering/leaving, and at k=0 anything seen on the belt is decoration
        conveyor = [] if gv.counter == 0 else [
            ConveyorBucket(s.color, s.fill, s.uncertain, (s.body,) if s.body else ())
            for s in gv.slots if not s.uncertain]
        return cls(grid, gv.lattice.exits(), list(gv.tappable()), list(gv.disagreements), locks,
                   gv.counter, conveyor_slots, conveyor, dict(gv.sand), dict(gv.exposed),
                   unknown_fill_need, dict(belt_stalled or {}), lower=dict(gv.lower),
                   above={c: dict(r) for c, r in gv.above.items()},
                   shells=list(getattr(gv, "shells", [])),
                   pic_locks=list(getattr(gv, "pic_locks", [])),
                   locked=getattr(gv, "locked", 0.0))

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
        """Buckets of unknown colour: one under every curtain cell and every lock cell, one per
        mystery bucket, N in a chest showing N (unread: 1), and opened cells not seen yet."""
        n = sum(len(lock.cells) for lock in self.locks
                if lock.kind in ("curtain", "lock", "mystery"))
        n += sum(lock.counter or 1 for lock in self.locks if lock.kind == "chest")
        # L190 tunnels: N more buckets each (unread number: at least 1)
        n += sum(1 if lock.counter is None else lock.counter for lock in self.locks
                 if lock.kind == "tunnel")
        return n + sum(c.hidden for c in self.cells(Kind.UNKNOWN))

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

    def stages_by_colour(self) -> Dict[str, float]:
        """Bucket stages still to fill per known colour: grid buckets (towers: both stages)
        and what the belt still needs."""
        out: Dict[str, float] = dict(self.grid_buckets())
        for b in self.conveyor:
            for col in {b.color, *b.stages} - {None}:
                out[col] = out.get(col, 0.0) + b.need_for(col, self.unknown_fill_need)
        return out

    def capacity_for(self, colour: Optional[str]) -> Optional[float]:
        """Sand one bucket of ``colour`` takes: its level-start measure (``colour_capacity``),
        else the level-wide estimate."""
        if colour and colour in self.colour_capacity:
            return self.colour_capacity[colour]
        return self.capacity_estimate()

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
        nxt.gates_opened, nxt.pics_opened = [], []
        nxt._open_chests()    # ... and chests already connected open anyway
        r, c = pos
        nxt.grid[r][c] = Cell(r, c, Kind.EMPTY)
        nxt.revealed = nxt._reveal_mysteries(pos)
        nxt.locks = [Lock(l.kind, l.unlock, max(0, l.counter - 1), l.cells)
                     if l.unlock == "taps" and l.counter is not None else l for l in nxt.locks]
        for lock in nxt.locks:
            for lr, lc in lock.cells:
                lcell = nxt.grid[lr][lc]
                if lcell is not None and lcell.kind is Kind.LOCK:
                    lcell.counter = lock.counter
        nxt._drop_from_tunnel(pos)
        nxt.unlocked = nxt._release_keys()
        nxt.chests_opened = nxt._open_chests()
        nxt.conveyor = self.conveyor + [
            ConveyorBucket(cell.color, 0, stages=(cell.body,) if cell.body else ())]
        nxt.conveyor_k = None if self.conveyor_k is None else self.conveyor_k + 1
        # predicted: a newly connected bucket flips face-up; cells that disagreed stay excluded
        nxt.disagreements = [p for p in self.disagreements if p != pos]
        nxt.tappable = [p for p in nxt.playable() if p not in nxt.disagreements]
        return nxt

    def _drop_from_tunnel(self, pos: Pos) -> None:
        """L190 (user): the bucket in front of a tunnel's opening was taken -> its next bucket
        drops into that cell (colour random, not visible: UNKNOWN here - the worst case, it
        blocks the cell again) and its number goes down by 1. An unread number counts as at
        least one bucket left; an empty tunnel stays (a wall)."""
        for lock in list(self.locks):
            if lock.kind != "tunnel" or lock.front() != tuple(pos) or lock.counter == 0:
                continue
            r, c = pos
            self.grid[r][c] = Cell(r, c, Kind.UNKNOWN, hidden=1,
                                   reason="tunnel drop: colour not seen yet")
            left = None if lock.counter is None else lock.counter - 1
            self.locks[self.locks.index(lock)] = Lock(lock.kind, lock.unlock, left, lock.cells,
                                                      lock.color, lock.opening)
            (tr, tc), = lock.cells
            if self.grid[tr][tc] is not None:
                self.grid[tr][tc].counter = left
            return

    def _release_keys(self) -> List[Tuple[str, Tuple[Pos, ...]]]:
        """Keys the path rule connects to the exit fly to the lock of their colour: the key's
        cell empties (confirmed, GAME_RULES.md) and the lock's cells open (buckets of unknown
        colour until seen: UNKNOWN here; a chain gate's cells become EMPTY, ``gates_opened``;
        L188: no grid lock of its colour but one picture lock -> it opens, ``pics_opened``).
        Repeats while a released key connects another key.
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
                targets = [l for l in self.locks if l.kind in ("lock", "chain")
                           and l.color == key.color]
                pics = [p for p in self.pic_locks if p.colour == key.color]
                if not targets and len(pics) == 1:   # L188: the picture lock of its colour
                    self.pic_locks.remove(pics[0])
                    self.pics_opened.append(key.color or "?")
                    continue
                if len(targets) != 1:
                    continue
                target = targets[0]
                self.locks.remove(target)
                if target.kind == "chain":   # a gate: its cells are free now (L189, user)
                    for lr, lc in target.cells:
                        self.grid[lr][lc] = Cell(lr, lc, Kind.EMPTY)
                    self.gates_opened.append((key.color or "?", target.cells))
                    continue
                for lr, lc in target.cells:
                    self.grid[lr][lc] = Cell(lr, lc, Kind.UNKNOWN,
                                             reason="opened lock: bucket colour not seen yet")
                opened.append((key.color or "?", target.cells))

    def _open_chests(self) -> List[Tuple[Pos, int]]:
        """Chests the path rule connects to the exit open and scatter N buckets of unknown colour
        into free cells (GAME_RULES.md, L186). Where they land is random: worst case they block
        paths, so the chest's cell stays an obstacle (UNKNOWN) and the strategy prices the
        scatter (planner.scatter_risk); the bot re-reads the whole grid after it."""
        conn = self.connected()
        opened = []
        for lock in [l for l in self.locks if l.kind == "chest" and l.cells[0] in conn]:
            (cr, cc), = lock.cells
            self.locks.remove(lock)
            n = lock.counter or 1
            self.grid[cr][cc] = Cell(cr, cc, Kind.UNKNOWN, hidden=n,
                                     reason=f"chest opened: {n} buckets scatter")
            opened.append(((cr, cc), n))
        return opened

    def _reveal_mysteries(self, pos: Pos) -> List[Pos]:
        """Tapping a bucket reveals the mystery buckets 4-adjacent to it: normal buckets from
        then on, colour not seen yet (UNKNOWN here)."""
        r, c = pos
        out = []
        for lock in [l for l in self.locks if l.kind == "mystery"]:
            (mr, mc), = lock.cells
            if abs(mr - r) + abs(mc - c) == 1:
                self.locks.remove(lock)
                self.grid[mr][mc] = Cell(mr, mc, Kind.UNKNOWN, hidden=1,
                                         reason="mystery revealed: colour not seen yet")
                out.append((mr, mc))
        return out

    def shell_near(self, colour: Optional[str], near: float) -> Optional[float]:
        """L188: depth (share of the screen height above the picture's bottom edge) of the
        lowest shell of ``colour`` in the picture if it is within ``near``, else None."""
        depths = [s.depth for s in self.shells if s.colour == colour]
        return min(depths) if depths and min(depths) <= near else None

    def behind_gate(self, gate: "Lock") -> List[Pos]:
        """Buckets the path rule would connect if the chain gate ``gate`` were open now."""
        before = self.connected()
        nxt = copy.deepcopy(self)
        for r, c in gate.cells:
            nxt.grid[r][c] = Cell(r, c, Kind.EMPTY)
        return sorted(p for p in nxt.connected() - before
                      if nxt.cell(p).kind is Kind.BUCKET)

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
    """4-adjacent lock cells of the same kind and colour form one lock (L183 has only
    single-cell ones; whether a lock can cover several cells is not confirmed yet). A chain gate
    (L189) spans its padlock cell and the chain cells next to it, which may show no colour: a
    colourless chain cell joins its neighbour and the group takes the padlock's colour."""
    out = [l for l in locks if l.kind not in ("lock", "chain")]
    pending = [l for l in locks if l.kind in ("lock", "chain")]
    while pending:
        group = [pending.pop(0)]
        grew = True
        while grew:
            grew = False
            colour = next((g.color for g in group if g.color), None)
            for l in list(pending):
                if l.kind == group[0].kind and (l.color == colour or None in (l.color, colour)) \
                        and any(abs(r - gr) + abs(c - gc) == 1
                                for (r, c) in l.cells for g in group for (gr, gc) in g.cells):
                    group.append(l)
                    pending.remove(l)
                    grew = True
        first = group[0]
        cells = tuple(sorted(p for g in group for p in g.cells))
        colour = next((g.color for g in group if g.color), None)
        out.append(Lock(first.kind, first.unlock, first.counter, cells, colour))
    return out


class BeltTracker:
    """Remembers the belt across frames: for every colour on the belt, how long its buckets'
    fill has not grown (a starving bucket = that colour's sand is not reachable right now).

    Per colour the reference is (number of buckets with a read fill < 100%, sum and max of their
    fills) at the last restart. The clock restarts when more buckets of the colour are read than
    the reference had (a new one arrived), when one is seen at 100% while others stay, or when
    the fills grew by ``min_growth`` points: the sum with as many buckets read, the max with
    fewer (run 161536: orange 0 % and orange/cream 38 % were read alternately one or both - the
    clock restarted every other frame and the stall never showed). The % labels jitter by one
    (99, 98, 99), which is no growth. Fewer buckets read for ``forget_s`` = one left unseen: the
    reference follows, the clock runs on. A colour that is not read in a frame (passing an edge,
    label unread) keeps its clock and stays reported for ``forget_s``; one seen with only
    finished buckets is dropped.

    With the k/5 counter (``k``) the bucket count is checked against it (run 20261021-122825,
    L190: two oranges read alternately one or both for minutes - after ``forget_s`` of one the
    reference followed down, the second one "arrived" again and the clock restarted every ~15 s,
    so the belt never counted as stuck): the reference follows fewer buckets only once k fell
    below its value at the reference (a bucket left unseen), and more buckets read restart the
    clock only if k rose since (a tap added one); otherwise the reference just takes the count.
    ``tracked`` = buckets the reference holds (seen recently), for k > buckets read this frame.
    """

    def __init__(self, min_growth: int = 2, forget_s: float = 3.0) -> None:
        self.min_growth, self.forget_s = min_growth, forget_s
        self._key: Dict[str, Tuple[int, int, int]] = {}    # colour -> (n, sum, max) reference
        self._since: Dict[str, float] = {}
        self._seen: Dict[str, float] = {}
        self._fewer: Dict[str, float] = {}                 # fewer buckets read than ref since
        self._k: Dict[str, Optional[int]] = {}             # counter k at the reference

    @property
    def tracked(self) -> int:
        return sum(n for n, _, _ in self._key.values())

    def update(self, conveyor: List[ConveyorBucket], now: float,
               k: Optional[int] = None) -> Dict[str, float]:
        """Feed one frame's belt (``k``: the k/5 counter, None if unread); returns colour ->
        seconds without fill growth (colours on the belt now, or out of sight for less than
        ``forget_s``)."""
        keys: Dict[str, Tuple[int, int, int]] = {}
        done = set()
        for b in conveyor:
            if b.color is None or b.fill is None:
                continue
            if b.fill >= 100:
                done.add(b.color)
                continue
            n, total, top = keys.get(b.color, (0, 0, 0))
            keys[b.color] = (n + 1, total + b.fill, max(top, b.fill))
        g = self.min_growth
        for color, (n, total, top) in keys.items():
            old = self._key.get(color)
            k0 = self._k.get(color)
            known = k is not None and k0 is not None
            if old is not None and n > old[0] and known and k <= k0:
                # an unread bucket of the colour is back, not a new one: no restart
                self._key[color] = (n, max(total, old[1]), max(top, old[2]))
                self._fewer.pop(color, None)
            elif (old is None or n > old[0] or color in done
                    or (n == old[0] and total >= old[1] + g) or (n < old[0] and top >= old[2] + g)):
                self._key[color] = (n, total, top)
                self._since[color] = now
                self._k[color] = k
                self._fewer.pop(color, None)
            elif n < old[0]:
                since = self._fewer.setdefault(color, now)
                left = k < k0 if known else True     # without the counter: time alone
                if now - since > self.forget_s and left:   # one left unseen: follow, clock on
                    self._key[color] = (n, total, top)
                    self._k[color] = k
                    del self._fewer[color]
            else:
                self._fewer.pop(color, None)
            self._seen[color] = now
        for color in list(self._key):
            if color not in keys and (color in done or now - self._seen[color] > self.forget_s):
                del self._key[color], self._since[color], self._seen[color]
                self._fewer.pop(color, None)
                self._k.pop(color, None)
        return {c: now - self._since[c] for c in self._key}
