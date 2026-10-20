"""Whole-level simulator for the level planner (levelplan.py; docs/GAME_RULES.md "Level plan").

Simplified, deterministic model of one level from one frame:

* **Picture** = per column a stack of sand cells bottom -> top (``cell_px`` blocks of the
  picture zone, classified against the palette's sand colours; empty picture skipped, so sand
  above an emptied spot has fallen). A belt bucket reaches the lowest ``contact`` cells of every
  column; what it takes is removed and the sand above falls down. Buried colours count: the
  whole picture is known (user, 2026-10-20: more sand of a colour appears once the covering
  layer is collected).
* **Capacity per colour**: cells of the colour / its bucket stages still to fill (the level is
  balanced per colour; the picture reading under-counts some colours - L187 cream 102 cells for
  5 stages vs orange 756 for 8 - so one capacity for all would starve them). ``LevelMemory``
  corrects it per colour from what earlier attempts of the board used.
* **Belt**: up to ``slots`` buckets; one lap = ``lap_s`` seconds. Per lap every belt bucket,
  in order, takes its colour from the contact layer at the speed measured on real runs
  (tools/belt_speed.py on belt.csv of runs 142035, 145310, 161536): ``fill_rate`` % of a
  bucket per second per contact-layer share above ``fill_min_exposed``, at most ``fill_max``
  %/s. Below that share a bucket gets (almost) nothing: L187 orange at 7 % of the contact layer
  did not fill in 30 s. A full stage leaves (a tower's next stage starts); a curtain counts the
  fills.
* **Shells** (L188, picture.py): a shell sits in its column's stack and falls with the sand
  (``Shell.below`` = sand cells under it). A shell bucket takes a slot but collects nothing
  until a shell of its colour is at the bottom (``below`` <= ``contact``); then the shell is
  used up and the bucket collects like any other. Shell buckets are tapped only when such a
  shell is at the bottom (the live rule, strategy.evaluate).
* **Picture locks** (L188): the boxes' cells are ``LOCKED`` sand that is never collected (the
  sand above rests on them). A key that opens one makes its sand unknown: the plan ends there
  ("horizon", as for curtains).
* **Grid**: ``LevelState.after_tap`` (path rule, keys -> locks, ice counts down by taps,
  chests, mystery buckets). Ice at 0 and curtains at 0 open to buckets of unknown colour
  (UNKNOWN cells): the plan cannot use them until they are seen (re-plan).

Actions: tap a tappable bucket of known colour (needs a free slot; ``tap_s`` seconds of belt
pass while the bot acts) or wait (laps until a bucket leaves the belt or nothing moves, at most
``max_wait_s``). Jam = belt full and nothing collects. ``Sim.t`` = belt seconds since the start
(the plan's predicted timing, compared with the real belt live).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from cells import Cell, Kind
from state import LevelState, Pos

Sig = Tuple[str, ...]
UNREAD = "?"                          # a belt bucket the reader did not see: keeps its slot
LOCKED = "locked"                     # picture cells behind a picture lock (L188)


@dataclass
class Shell:
    colour: str
    col: int                          # picture column (stack index)
    below: int                        # sand cells under it in its column


@dataclass
class Picture:
    names: List[str]
    cols: List[List[int]]             # per column: colour indices bottom -> top
    contact: int                      # cells a bucket reaches per column
    shells: List[Shell] = field(default_factory=list)

    def copy(self) -> "Picture":
        return Picture(self.names, [list(c) for c in self.cols], self.contact,
                       [Shell(s.colour, s.col, s.below) for s in self.shells])

    def shell_down(self, colour: str) -> Optional[Shell]:
        """A shell of ``colour`` at the bottom (it drops into a waiting shell bucket)."""
        return next((s for s in self.shells if s.colour == colour and s.below <= self.contact),
                    None)

    def total(self) -> int:
        return sum(len(c) for c in self.cols)

    def count(self, colour: str) -> int:
        if colour not in self.names:
            return 0
        i = self.names.index(colour)
        return sum(c.count(i) for c in self.cols)

    def exposed(self, colour: str) -> int:
        if colour not in self.names:
            return 0
        i = self.names.index(colour)
        return sum(c[:self.contact].count(i) for c in self.cols)

    def share(self, colour: str) -> float:
        """The colour's share of the contact layer (as vision's ``exposed``)."""
        n = sum(min(self.contact, len(c)) for c in self.cols)
        return self.exposed(colour) / n if n else 0.0

    def take(self, colour: str, n: int) -> int:
        """Remove up to ``n`` cells of ``colour`` from the contact layer, lowest first."""
        if n <= 0 or colour not in self.names:
            return 0
        i, got = self.names.index(colour), 0
        for k, col in enumerate(self.cols):
            j = 0
            while j < min(self.contact, len(col)) and got < n:
                if col[j] == i:
                    del col[j]          # the sand above falls down (with its shells)
                    got += 1
                    for s in self.shells:
                        if s.col == k and s.below > j:
                            s.below -= 1
                else:
                    j += 1
            if got >= n:
                break
        return got


def picture_from_frame(img: np.ndarray, palette, v, cell_px: int = 8, shells=(),
                       pic_locks=()) -> Picture:
    """Stacks from the picture of a gameplay frame, cut at its real bottom edge (as vision's
    contact layer): per ``cell_px`` block the most frequent sand colour of its pixels (empty
    picture when most pixels are empty / not sand). ``shells`` / ``pic_locks``: picture.py
    objects of the frame (L188): shells become ``Shell`` markers, lock boxes ``LOCKED`` cells."""
    from palette import bgr_to_lab, lab_distance
    from sand import _bottom_edge

    h, w = img.shape[:2]
    x0, y0, x1, y1 = (int(v.picture[0] * w), int(v.picture[1] * h), int(v.picture[2] * w),
                      int(v.picture[3] * h))
    edge = _bottom_edge(img, palette, v)
    pic = img[y0:edge if edge is not None else y1, x0:x1]
    rows, ncols = pic.shape[0] // cell_px, pic.shape[1] // cell_px
    pic = pic[pic.shape[0] - rows * cell_px:, :ncols * cell_px]     # whole blocks, bottom kept
    lab = bgr_to_lab(pic)
    idx, names = palette.classify_pixels(lab, "sand", v.sand_max_dist)
    if "picture_empty" in palette.ui:
        idx[lab_distance(lab, palette.ui["picture_empty"]) < 10] = -1
    blocks = idx.reshape(rows, cell_px, ncols, cell_px).transpose(0, 2, 1, 3)
    blocks = blocks.reshape(rows, ncols, cell_px * cell_px) + 1          # 0 = empty
    counts = np.stack([(blocks == i).sum(axis=2) for i in range(len(names) + 1)], axis=2)
    cell = counts.argmax(axis=2) - 1
    names = list(names)
    top = (edge if edge is not None else y1) - rows * cell_px    # frame row of block row 0
    if pic_locks:
        names.append(LOCKED)
        for lk in pic_locks:
            bx0, by0, bx1, by1 = lk.rect
            r0, r1 = int((by0 * h - top) // cell_px), int(-(-(by1 * h - top) // cell_px))
            c0, c1 = int((bx0 * w - x0) // cell_px), int(-(-(bx1 * w - x0) // cell_px))
            cell[max(0, r0):max(0, r1), max(0, c0):max(0, c1)] = len(names) - 1
    cols = [[int(x) for x in cell[::-1, j] if x >= 0] for j in range(ncols)]
    contact = max(1, round(v.contact_layer * h / cell_px))
    markers = []
    for sh in shells:
        j = int((sh.x * w - x0) // cell_px)
        if sh.colour is None or not 0 <= j < ncols:
            continue
        r = int((sh.y * h - top) // cell_px)        # block row of the shell's centre
        below = int((cell[max(0, r + 1):, j] >= 0).sum())
        markers.append(Shell(sh.colour, j, below))
    return Picture(names, cols, contact, markers)


@dataclass
class Slot:
    stages: Tuple[str, ...]           # colours top first
    left: int                         # cells the current stage still needs
    stage: int = 0
    credit: float = 0.0               # cells collected but not yet whole
    shell: bool = False               # L188 shell bucket still waiting for its shell

    @property
    def colour(self) -> str:
        return self.stages[self.stage]


@dataclass
class SimConfig:
    slots: int = 5
    lap_s: float = 1.0                # belt seconds per lap
    fill_rate: float = 100.0          # % of a bucket per s per contact share above ...
    fill_min_exposed: float = 0.08    # ... this share (below: no fill)
    fill_max: float = 30.0            # % of a bucket per s at most
    tap_s: float = 4.2                # belt seconds that pass per tap of the bot
    max_wait_s: float = 60.0

    @property
    def laps_per_tap(self) -> int:
        return max(0, round(self.tap_s / self.lap_s))

    def rate(self, share: float) -> float:
        """% of a bucket per second at this contact-layer share."""
        return max(0.0, min(self.fill_max, self.fill_rate * (share - self.fill_min_exposed)))


@dataclass
class Sim:
    grid: LevelState                  # grid, locks (conveyor fields unused)
    belt: List[Slot]
    pic: Picture
    capacity: Dict[str, int]          # cells per bucket stage, per colour
    sc: SimConfig
    taps: List[Pos] = field(default_factory=list)
    actions: List[Tuple[str, Optional[Pos], Optional[str], int]] = field(default_factory=list)
    # ^ ("tap" | "wait", pos, colour, belt buckets before the action)
    fills: int = 0                    # bucket stages completed
    collected: int = 0
    jam: bool = False
    t: float = 0.0                    # belt seconds since the start
    full_taps: int = 0                # taps that took the last free slot
    pics_opened: int = 0              # picture locks opened by keys in the simulation

    # -- construction ---------------------------------------------------------------------

    @classmethod
    def start(cls, state: LevelState, pic: Picture, sc: SimConfig,
              capacity: Optional[Dict[str, int]] = None) -> "Sim":
        """From a live state: its grid, its belt (fill -> cells left) and the picture.
        ``capacity``: cells per stage for some colours (memory); the rest is balanced."""
        grid = copy.deepcopy(state)
        grid.sand, grid.exposed, grid.above, grid.lower = {}, {}, {}, {}
        grid.belt_stalled = {}
        stages: Dict[str, float] = {}
        for c in grid.cells(Kind.BUCKET):
            for col in (c.color, c.body):
                if col:
                    stages[col] = stages.get(col, 0.0) + 1
        for b in state.conveyor:
            for col in (b.color,) + tuple(b.stages):
                if col:
                    stages[col] = stages.get(col, 0.0) + (
                        b.need(state.unknown_fill_need) if col == b.color else 1)
        cap = {col: max(1, round(pic.count(col) / n)) for col, n in stages.items() if n > 0}
        cap.update(capacity or {})
        belt = []
        for b in state.conveyor:
            if not b.color or (b.fill is not None and b.fill >= 100):
                continue
            need = b.need(state.unknown_fill_need)
            belt.append(Slot((b.color,) + tuple(b.stages),
                             max(1, round(need * cap.get(b.color, 1)))))
        # buckets the counter shows but the belt reader did not (off view, label unread): a
        # slot that stays taken (run 171133 551: k 4, 3 read - the plan thought 2 were free)
        unread = sum(1 for b in state.conveyor if not b.color) + max(
            0, (state.conveyor_k or 0) - len(state.conveyor))
        belt += [Slot((UNREAD,), 1) for _ in range(unread)]
        return cls(grid, belt, pic, cap, sc)

    def clone(self) -> "Sim":
        return Sim(self.grid, [Slot(s.stages, s.left, s.stage, s.credit, s.shell)
                               for s in self.belt],
                   self.pic.copy(), self.capacity, self.sc, list(self.taps),
                   list(self.actions), self.fills, self.collected, self.jam, self.t,
                   self.full_taps, self.pics_opened)

    # -- queries --------------------------------------------------------------------------

    @property
    def free(self) -> int:
        return self.sc.slots - len(self.belt)

    def moves(self) -> List[Pos]:
        """Tappable buckets of known colour; a shell bucket only with its shell at the bottom
        (L188, the live rule)."""
        if self.free <= 0:
            return []
        out = []
        for p in self.grid.tappable:
            c = self.grid.cell(p)
            if c.kind is Kind.BUCKET and c.color and (
                    not c.shell or self.pic.shell_down(c.color) is not None):
                out.append(p)
        return out

    def dead_slots(self) -> int:
        """Belt buckets that can never fill: not enough of their colour left in the picture."""
        need: Dict[str, int] = {}
        for s in self.belt:
            need[s.colour] = need.get(s.colour, 0) + s.left
        return sum(1 for s in self.belt
                   if s.colour != UNREAD and self.pic.count(s.colour) < need[s.colour])

    def stuck_slots(self) -> int:
        """Belt buckets that collect nothing now (their colour too little in the contact layer)."""
        return sum(1 for s in self.belt
                   if s.colour != UNREAD and self.sc.rate(self.pic.share(s.colour)) <= 0)

    def belt_fills(self) -> List[Tuple[str, int]]:
        """(colour, fill %) of the belt buckets, as the live belt reader shows them."""
        return [(s.colour, int(100 * (1 - s.left / max(1, self.capacity.get(s.colour, 1)))))
                for s in self.belt]

    @property
    def won(self) -> bool:
        """Every bucket filled (the sand of colours without a known bucket may stay)."""
        return (not self.belt and not self.grid.cells(Kind.BUCKET)
                and not self.unknown_left())

    def unknown_left(self) -> int:
        """Cells whose content is not known (opened locks / curtains / ice ...) + 1 per picture
        lock a key opened in the simulation (its sand's colours are unknown, L188)."""
        return len(self.grid.cells(Kind.UNKNOWN)) + sum(
            len(l.cells) for l in self.grid.locks if l.kind in ("curtain", "lock", "ice")) + \
            self.pics_opened

    def signature(self) -> Sig:
        return grid_signature(self.grid.grid)

    # -- actions --------------------------------------------------------------------------

    def tap(self, pos: Pos) -> "Sim":
        nxt = self.clone()
        cell = self.grid.cell(pos)
        nxt.actions.append(("tap", pos, cell.color, len(self.belt)))
        g = self.grid.after_tap(pos)
        g.conveyor, g.conveyor_k = [], 0
        nxt.grid = g
        nxt.pics_opened += len(g.pics_opened)
        nxt._open_ice()
        nxt.belt.append(Slot((cell.color,) + ((cell.body,) if cell.body else ()),
                             nxt.capacity.get(cell.color, 1), shell=cell.shell))
        nxt.taps.append(pos)
        nxt.full_taps += nxt.free == 0
        for _ in range(self.sc.laps_per_tap):
            nxt.lap()
        return nxt

    def wait(self) -> Optional["Sim"]:
        """Laps until a bucket leaves the belt; None if nothing moved at all (a useless wait).
        A full belt that stops moving is a jam."""
        nxt = self.clone()
        nxt.actions.append(("wait", None, None, len(self.belt)))
        n0, moved = len(nxt.belt), 0.0
        for _ in range(max(1, round(self.sc.max_wait_s / self.sc.lap_s))):
            got = nxt.lap()
            moved += got
            if len(nxt.belt) < n0 or got <= 0:
                break
        if moved == 0:
            if nxt.free == 0:
                nxt.jam = True
                return nxt
            return None
        return nxt

    def lap(self) -> float:
        """One lap; returns the cells the belt earned (0: nothing collects)."""
        got = 0.0
        self.t += self.sc.lap_s
        for s in list(self.belt):
            if s.shell:   # L188: nothing until a shell of its colour drops in
                down = self.pic.shell_down(s.colour)
                if down is None:
                    continue
                self.pic.shells.remove(down)
                s.shell = False
            rate = self.sc.rate(self.pic.share(s.colour))
            if rate <= 0:
                continue
            s.credit += rate / 100 * self.capacity.get(s.colour, 1) * self.sc.lap_s
            want = min(s.left, int(s.credit))
            n = self.pic.take(s.colour, want)
            if n < want:                  # the contact layer ran out of the colour
                s.credit = 0.0
                got += n
            else:
                s.credit -= n
                got += max(n, 1e-3)       # collecting (a fraction of a cell this lap)
            s.left -= n
            self.collected += n
            if s.left <= 0:
                self.fills += 1
                self._fill_curtains()
                if s.stage + 1 < len(s.stages):
                    s.stage += 1
                    s.left, s.credit = self.capacity.get(s.colour, 1), 0.0
                else:
                    self.belt.remove(s)
        return got

    def _open_ice(self) -> None:
        g = self.grid
        for lock in [l for l in g.locks if l.kind == "ice" and l.counter == 0]:
            g.locks.remove(lock)
            self._unknown(lock.cells, "ice opened")

    def _fill_curtains(self) -> None:
        g = self.grid
        from state import Lock
        for lock in list(g.locks):
            if lock.unlock == "fills" and lock.counter is not None:
                g.locks.remove(lock)
                if lock.counter <= 1:
                    self._unknown(lock.cells, "curtain opened")
                else:
                    g.locks.append(Lock(lock.kind, lock.unlock, lock.counter - 1, lock.cells,
                                        lock.color))

    def _unknown(self, cells, why: str) -> None:
        g = self.grid
        for r, c in cells:
            g.grid[r][c] = Cell(r, c, Kind.UNKNOWN, hidden=1, reason=f"{why}: colour not seen")
        g.tappable = [p for p in g.playable() if p not in g.disagreements]


def grid_signature(grid) -> Sig:
    """What the plan predicts of the board and the screen can confirm: per cell empty, bucket
    colour(s), lock kind / colour or unknown (not the face: it flips by the path rule)."""
    out = []
    for row in grid:
        for cell in row:
            if cell is None:
                out.append("x")
            elif cell.kind is Kind.EMPTY:
                out.append(".")
            elif cell.kind is Kind.BUCKET:
                out.append((f"{cell.color}/{cell.body}" if cell.body else str(cell.color))
                           + ("+shell" if cell.shell else ""))
            elif cell.kind is Kind.LOCK:
                out.append(f"{cell.lock}:{cell.color or ''}")
            else:
                out.append("?")
    return tuple(out)
