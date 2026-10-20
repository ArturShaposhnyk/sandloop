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
* **Belt**: up to ``slots`` buckets; one lap = every belt bucket, in order, takes up to
  ``lap_take`` cells of its colour from the contact layer. A full stage leaves (a tower's next
  stage starts); a curtain counts the fills.
* **Grid**: ``LevelState.after_tap`` (path rule, keys -> locks, ice counts down by taps,
  chests, mystery buckets). Ice at 0 and curtains at 0 open to buckets of unknown colour
  (UNKNOWN cells): the plan cannot use them until they are seen (re-plan).

Actions: tap a tappable bucket of known colour (needs a free slot; ``laps_per_tap`` laps pass
while the bot acts) or wait (laps until a bucket leaves the belt or nothing moves). Jam = belt
full and a lap moves nothing.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from cells import Cell, Kind
from state import LevelState, Pos

Sig = Tuple[str, ...]


@dataclass
class Picture:
    names: List[str]
    cols: List[List[int]]             # per column: colour indices bottom -> top
    contact: int                      # cells a bucket reaches per column

    def copy(self) -> "Picture":
        return Picture(self.names, [list(c) for c in self.cols], self.contact)

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

    def take(self, colour: str, n: int) -> int:
        """Remove up to ``n`` cells of ``colour`` from the contact layer, lowest first."""
        if n <= 0 or colour not in self.names:
            return 0
        i, got = self.names.index(colour), 0
        for col in self.cols:
            j = 0
            while j < min(self.contact, len(col)) and got < n:
                if col[j] == i:
                    del col[j]          # the sand above falls down
                    got += 1
                else:
                    j += 1
            if got >= n:
                break
        return got


def picture_from_frame(img: np.ndarray, palette, v, cell_px: int = 8) -> Picture:
    """Stacks from the picture zone of a gameplay frame (block centres, nearest sand colour)."""
    from palette import bgr_to_lab, lab_distance
    from zones import crop_rel

    pic = crop_rel(img, v.picture)
    h, w = img.shape[:2]
    half = cell_px // 2
    lab = bgr_to_lab(pic[half::cell_px, half::cell_px])
    idx, names = palette.classify_pixels(lab, "sand", v.sand_max_dist)
    if "picture_empty" in palette.ui:
        idx[lab_distance(lab, palette.ui["picture_empty"]) < 10] = -1
    cols = [[int(x) for x in idx[::-1, j] if x >= 0] for j in range(idx.shape[1])]
    contact = max(1, round(v.contact_layer * h / cell_px))
    return Picture(list(names), cols, contact)


@dataclass
class Slot:
    stages: Tuple[str, ...]           # colours top first
    left: int                         # cells the current stage still needs
    stage: int = 0

    @property
    def colour(self) -> str:
        return self.stages[self.stage]


@dataclass
class SimConfig:
    slots: int = 5
    lap_take: int = 20                # cells a bucket takes per lap at most
    laps_per_tap: int = 1
    max_wait_laps: int = 60


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
        return cls(grid, belt, pic, cap, sc)

    def clone(self) -> "Sim":
        return Sim(self.grid, [Slot(s.stages, s.left, s.stage) for s in self.belt],
                   self.pic.copy(), self.capacity, self.sc, list(self.taps),
                   list(self.actions), self.fills, self.collected, self.jam)

    # -- queries --------------------------------------------------------------------------

    @property
    def free(self) -> int:
        return self.sc.slots - len(self.belt)

    def moves(self) -> List[Pos]:
        if self.free <= 0:
            return []
        return [p for p in self.grid.tappable
                if self.grid.cell(p).kind is Kind.BUCKET and self.grid.cell(p).color]

    def dead_slots(self) -> int:
        """Belt buckets that can never fill: not enough of their colour left in the picture."""
        need: Dict[str, int] = {}
        for s in self.belt:
            need[s.colour] = need.get(s.colour, 0) + s.left
        return sum(1 for s in self.belt if self.pic.count(s.colour) < need[s.colour])

    def stuck_slots(self) -> int:
        """Belt buckets whose colour is not in the contact layer now (they wait for sand)."""
        return sum(1 for s in self.belt if self.pic.exposed(s.colour) == 0)

    @property
    def won(self) -> bool:
        """Every bucket filled (the sand of colours without a known bucket may stay)."""
        return (not self.belt and not self.grid.cells(Kind.BUCKET)
                and not self.unknown_left())

    def unknown_left(self) -> int:
        return len(self.grid.cells(Kind.UNKNOWN)) + sum(
            len(l.cells) for l in self.grid.locks if l.kind in ("curtain", "lock", "ice"))

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
        nxt._open_ice()
        nxt.belt.append(Slot((cell.color,) + ((cell.body,) if cell.body else ()),
                             nxt.capacity.get(cell.color, 1)))
        nxt.taps.append(pos)
        for _ in range(self.sc.laps_per_tap):
            nxt.lap()
        return nxt

    def wait(self) -> Optional["Sim"]:
        """Laps until a bucket leaves the belt; None if nothing moved at all (a useless wait).
        A full belt that stops moving is a jam."""
        nxt = self.clone()
        nxt.actions.append(("wait", None, None, len(self.belt)))
        n0, moved = len(nxt.belt), 0
        for _ in range(self.sc.max_wait_laps):
            got = nxt.lap()
            moved += got
            if len(nxt.belt) < n0 or got == 0:
                break
        if moved == 0:
            if nxt.free == 0:
                nxt.jam = True
                return nxt
            return None
        return nxt

    def lap(self) -> int:
        got = 0
        for s in list(self.belt):
            n = self.pic.take(s.colour, min(s.left, self.sc.lap_take))
            s.left -= n
            got += n
            if s.left <= 0:
                self.fills += 1
                self._fill_curtains()
                if s.stage + 1 < len(s.stages):
                    s.stage += 1
                    s.left = self.capacity.get(s.colour, 1)
                else:
                    self.belt.remove(s)
        self.collected += got
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
                out.append(f"{cell.color}/{cell.body}" if cell.body else str(cell.color))
            elif cell.kind is Kind.LOCK:
                out.append(f"{cell.lock}:{cell.color or ''}")
            else:
                out.append("?")
    return tuple(out)
