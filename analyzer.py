"""Frame -> decision, shared by bot.py, the overlay and ``python -m strategy``: ``Analyzer``
reads a frame (screen type, grid, belt) and runs ``strategy.choose``; ``DecisionGate`` confirms a
move over consecutive frames. Split out of strategy.py; never taps."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from cells import Kind
from state import BeltTracker, LevelState, Pos
from strategy import Candidate, Decision, choose

if TYPE_CHECKING:
    import numpy as np

    from vision import GameView



@dataclass
class Analysis:
    screen: str
    view: Optional["GameView"] = None
    state: Optional[LevelState] = None
    decision: Optional[Decision] = None
    error: Optional[str] = None
    failure: Optional[str] = None      # failed read of a gameplay frame: "<step>: <reason>"
    reads: int = 0                     # gameplay frames read so far by this Analyzer
    fails: int = 0                     # ... and how many of them failed
    screen_result: Any = None          # screens.ScreenResult (anchors found, for flow.py)
    speed_x2: Optional[bool] = None    # belt speed 2x, confirmed this level (None: not yet)
    speed_read: Optional[bool] = None  # ... this frame's reading of the ▶▶2x button

    def tap_rel(self) -> Optional[Tuple[float, float]]:
        """Relative screen point of the chosen move (what a tap would hit), or None."""
        if self.view is None or self.decision is None or self.decision.move is None:
            return None
        return self.view.cell_center_rel(*self.decision.move.pos)


class Analyzer:
    """Loads palette, digits and screen templates once; ``analyze(frame)`` reads a frame of the
    iPhone screen and decides the next move. Never taps. ``clock``: the default frame time
    (belt stall clock) when ``analyze`` gets no ``now``; tests pass a fake one."""

    def __init__(self, cfg, clock=None) -> None:
        import time

        from digits import DigitReader
        from palette import Palette
        from screens import ScreenClassifier, load_templates
        from hints import HintArrows
        from icons import IconMatcher

        self.cfg = cfg
        self.clock = clock or time.monotonic
        self.palette = Palette.load(cfg.paths.palette)
        self.reader = DigitReader.load(Path(cfg.paths.dataset) / "digits.npz")
        self.icons = IconMatcher.load(Path(cfg.paths.dataset) / "templates" / "icons",
                                      self.palette)
        self.hints = HintArrows.load(Path(cfg.paths.dataset) / "templates" / "hints")
        self.reset_level(track=False)
        self.classifier = ScreenClassifier(load_templates(Path(cfg.paths.dataset) / "templates"),
                                           cfg.vision)
        self.belt = BeltTracker(cfg.strategy.stall_min_growth, cfg.strategy.stall_forget_s)
        self.fail_dir = Path(cfg.paths.debug) / "vision_fail"
        self.colour_dir = Path(cfg.paths.dataset) / "unknown_colours"
        self.unknown_dir = Path(cfg.paths.dataset) / "unknown"
        self.colours_seen: set = set()                  # unknown colours saved this run
        self.cells_seen: set = set()                    # unknown cells (pos, reason) saved
        self.reads = self.fails = self.fails_saved = 0

    def reset_level(self, track: bool = True) -> None:
        """Forget what belongs to one level. ``track``: the frames that follow are one level
        played live (bot.py, overlay): keep its grid geometry (hints.LatticeLock) and watch for
        chests opening. Off for unrelated screenshots (tests, ``python -m strategy``)."""
        from hints import CellMemory, InFlight, LatticeLock, RecentUnlocks

        cfg = self.cfg
        self.memory = CellMemory()   # last clean cell readings, for cells under hint arrows
        self.unlocks = RecentUnlocks(cfg.vision.unlock_trust_s)   # key locks that just opened
        self.lattices: Optional[LatticeLock] = (
            LatticeLock(cfg.vision.lattice_lock_frames, cfg.vision.lattice_min_fit) if track
            else None)
        self.flights: Optional[InFlight] = InFlight(cfg.vision.flight_s) if track else None
        self.planner = None                    # levelplan.PlanRunner (live, planner enabled)
        if track and cfg.planner.enabled:
            from levelplan import PlanRunner

            self.planner = PlanRunner(cfg, Path(cfg.paths.debug) / "levels.jsonl")
        self.colour_capacity: Optional[Dict[str, float]] = None   # measured on the 1st read
        self.chests: Dict[Pos, int] = {}       # chests in the last read (cell -> N)
        self.chest_log: List[Dict[str, Any]] = []   # chests opened this level + where their
        #   buckets landed (levels.jsonl: are chest results the same on Try Again?)
        self._labels: Optional[List[List[str]]] = None   # last read's labels
        self.scatter_until = -1.0              # a chest opened: no decisions until then
        self.mechanics_noted: set = set()      # "unknown mechanic" notes logged this level
        self.hiding: Optional[set] = None      # cells under ice / key locks / curtains at start
        self.revealed: Dict[str, str] = {}     # "r,c" -> colour seen there once open (memory)
        self._reveal_seen: Dict[Pos, str] = {}
        from hud import SpeedTracker

        self.speed = SpeedTracker(cfg.bot.speed_confirm_frames)   # ▶▶2x, per level

    def analyze(self, img: "np.ndarray", log=print, now: Optional[float] = None,
                diff: Optional[float] = None, adjust=None) -> Analysis:
        """``now``: frame time in seconds (default: ``self.clock()``) for the belt stall clock.
        ``diff``: frame-to-frame change at capture, logged with failed reads. ``adjust(state)``
        may amend the state before the decision (bot.py: what it knows of released towers)."""
        from vision import VisionError, read_gameplay

        from screens import ScreenResult, popup_bar

        now = self.clock() if now is None else now
        found = self.classifier.classify(img)
        screen = found.screen
        if screen == "gameplay" and popup_bar(img, self.cfg.vision):
            # a popup sliding in (revive): the board under it is not read (run 101655 00507)
            screen, found = "popup", ScreenResult("popup", found.matches)
        if screen != "gameplay":
            if self.lattices is not None:
                self.lattices.unlock()   # another level may follow
            return Analysis(screen, reads=self.reads, fails=self.fails, screen_result=found)
        self.reads += 1
        from hud import read_speed

        speed_read = read_speed(img, self.cfg.bot)   # ▶▶2x button (bot.py taps it when off)
        if self.speed.feed(speed_read):
            log(f"belt speed 2x: {'on' if self.speed.state else 'off'}")
            if self.planner is not None:
                self.planner.set_speed(bool(self.speed.state), log)
        try:
            gv = read_gameplay(img, self.palette, self.cfg.vision, self.reader, log=log,
                               icons=self.icons, hints=self.hints, memory=self.memory,
                               unlocks=self.unlocks, now=now, lattices=self.lattices,
                               flights=self.flights)
        except VisionError as e:
            res = Analysis(screen, error=str(e), failure=f"grid: {e}")
        else:
            self._save_unknowns(img, gv, log)
            st = self.cfg.strategy
            state = LevelState.from_view(gv, st.conveyor_slots, st.unknown_fill_need)
            if self.lattices is not None:   # live level: per-colour capacity from its start
                if self.colour_capacity is None:
                    self.colour_capacity = colour_capacity(state)
                    log("capacity per colour (share of the picture per bucket): " + ", ".join(
                        f"{c} {v:.2%}" for c, v in sorted(self.colour_capacity.items())))
                state.colour_capacity = dict(self.colour_capacity)
            state.belt_stalled = self.belt.update(state.conveyor, now, state.conveyor_k)
            state.belt_tracked = self.belt.tracked
            self._note_mechanics(state, log)
            if self.lattices is not None:
                self._watch_hidden(state, log)
            if adjust is not None:
                adjust(state)
            res = Analysis(screen, gv, state, choose(state, st))
            if self.lattices is not None:
                self._chest_scatter(gv, res, now, log)
            if self.planner is not None and now >= self.scatter_until:
                res.decision = self.planner.decide(img, gv, state, res.decision, now, log)
            if gv.counter is None:
                res.failure = "counter: k/5 not read"
        if res.failure:
            self.fails += 1
            self._save_failure(img, res.failure, diff, log)
        res.reads, res.fails = self.reads, self.fails
        res.speed_x2, res.speed_read = self.speed.state, speed_read
        return res

    def _watch_hidden(self, state: LevelState, log) -> None:
        """Level memory (levelmemory.LevelMemory.revealed): the colour of a bucket that was
        under ice / a key lock / a curtain at the level's start, once read the same twice in a
        row (not hidden by an arrow, not disagreeing with the path rule). Try Again restarts the
        same board, so the planner can plan past that lock next time."""
        from cells import Kind

        if self.hiding is None:
            self.hiding = {(c.row, c.col) for c in state.cells(Kind.LOCK)
                           if c.lock in ("ice", "lock", "curtain")}
        for pos in self.hiding:
            key = f"{pos[0]},{pos[1]}"
            cell = state.cell(pos)
            if key in self.revealed or cell is None:
                continue
            if (cell.kind is not Kind.BUCKET or not cell.color or cell.occluded
                    or pos in state.disagreements):
                self._reveal_seen.pop(pos, None)
                continue
            label = cell.color + (f"/{cell.body}" if cell.body else "")
            if self._reveal_seen.get(pos) == label:
                self.revealed[key] = label
                log(f"r{pos[0]}c{pos[1]}: {label} under its lock (level memory)")
            self._reveal_seen[pos] = label

    def _note_mechanics(self, state: LevelState, log) -> None:
        """Once per level: what the bot cannot explain on this board yet (user, 2026-10-21:
        log "unknown mechanic" and ask instead of guessing)."""
        hidden = any(l.kind in ("curtain", "lock", "ice", "chest", "mystery")
                     for l in state.locks) or bool(state.cells(Kind.UNKNOWN))
        for pl in state.pic_locks:
            keys = [l for l in state.locks if l.kind == "key" and l.color == pl.colour]
            if keys:
                continue
            note = (f"picture lock {pl.colour}: its key is not on the board yet"
                    + (" (under a curtain / lock / ice?)" if hidden else
                       " and nothing hides one - unknown mechanic, ask the user"))
            if note not in self.mechanics_noted:
                self.mechanics_noted.add(note)
                log(note)
        shells = {c.color for c in state.cells(Kind.BUCKET) if c.shell}
        for colour in sorted(shells - {s.colour for s in state.shells}):
            note = (f"shell buckets {colour}: no {colour} shell seen in the picture (hidden behind "
                    "a picture lock / missed by the reader?) - they wait")
            if note not in self.mechanics_noted:
                self.mechanics_noted.add(note)
                log(note)

    def _chest_scatter(self, gv: "GameView", res: Analysis, now: float, log) -> None:
        """A chest that was in the last read and is gone has opened: its buckets scatter over
        the grid (GAME_RULES.md, L186). Forget the old cell readings (hint-arrow memory) and
        decide nothing for ``chest_settle_s``: every frame is a full re-read anyway."""
        chests = {(c.row, c.col): c.counter or 1 for row in gv.grid for c in row
                  if c is not None and c.kind is Kind.LOCK and c.lock == "chest"}
        opened = {p: n for p, n in self.chests.items() if p not in chests}
        self.chests = chests
        labels = [row.split() for row in gv.labels()]
        pending = next((e for e in self.chest_log if "landed" not in e), None)
        if pending is not None and now >= self.scatter_until:
            before = pending.pop("before")
            pending["landed"] = [[r, c, lab] for r, row in enumerate(labels)
                                 for c, lab in enumerate(row)
                                 if r < len(before) and c < len(before[r])
                                 and before[r][c] == "." and lab != "."]
            log("chest result: " + (", ".join(f"r{r}c{c} {lab}" for r, c, lab in
                                              pending["landed"]) or "no bucket landed"))
        if opened:
            self.chest_log.append({"t": round(now, 2), "before": self._labels or labels,
                                   "chests": [[r, c, n] for (r, c), n in opened.items()]})
            from hints import CellMemory

            self.memory = CellMemory()
            self.scatter_until = now + self.cfg.strategy.chest_settle_s
            if self.flights is not None:
                self.flights.pause(self.scatter_until + self.cfg.vision.flight_s)
            log("chest " + ", ".join(f"r{r}c{c} ({n})" for (r, c), n in opened.items())
                + " opened: buckets scatter - re-reading the whole grid")
        self._labels = labels
        if now < self.scatter_until:
            res.decision = Decision(None, "wait: a chest opened, its buckets are landing",
                                    res.decision.candidates if res.decision else [])

    def _save_unknowns(self, img: "np.ndarray", gv: "GameView", log) -> None:
        """New unknown colours -> dataset/unknown_colours/, new UNKNOWN cells (other reasons)
        -> dataset/unknown/; each once per run."""
        from unknown_colours import save_unknown, save_unknown_colours

        save_unknown_colours(img, gv, self.cfg.vision, self.colour_dir, self.colours_seen, log)
        cells = {(c.row, c.col, c.reason) for c in gv.unknown_cells() if c.lab is None}
        if cells - self.cells_seen:
            self.cells_seen |= cells
            save_unknown(img, gv, self.unknown_dir, log)

    def _save_failure(self, img: "np.ndarray", failure: str, diff: Optional[float],
                      log) -> None:
        """Frame + step to debug/vision_fail/ (and fails.jsonl), to find the pattern behind
        misreads. Capped at strategy.vision_fail_max_saves per run (disk)."""
        import json
        import time

        import cv2

        if self.fails_saved >= self.cfg.strategy.vision_fail_max_saves:
            return
        self.fail_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
        step = failure.split(":")[0]
        path = self.fail_dir / f"{stamp}-{step}.png"
        if not cv2.imwrite(str(path), img):
            log(f"vision_fail: cannot write {path}")
            return
        self.fails_saved += 1
        rec = {"time": stamp, "frame": path.name, "step": step, "failure": failure,
               "diff": None if diff is None else round(diff, 5),
               "reads": self.reads, "fails": self.fails}
        with (self.fail_dir / "fails.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        if self.fails_saved == self.cfg.strategy.vision_fail_max_saves:
            log(f"vision_fail: saved {self.fails_saved} frames, not saving more this run")


def colour_capacity(state: LevelState) -> Dict[str, float]:
    """Per colour: its sand / its bucket stages still to fill. Levels are balanced per colour,
    and this ratio cancels the reading's bias per colour (L187: darkred read 1.0 % of the picture
    while it was 39 % of the contact layer, so the shared estimate rejected both darkred buckets
    for "not enough sand" to the end of run 20261020-145310). Measured once per level, at its
    start; colours seen later (opened locks) use the level-wide estimate."""
    return {c: state.sand[c] / n for c, n in state.stages_by_colour().items()
            if n > 0 and state.sand.get(c, 0.0) > 0}


class DecisionGate:
    """A move is confirmed only after ``frames`` consecutive analyses chose the same bucket
    (cell and colour) with no failed read in between; anything else restarts the count. The
    bot taps only confirmed moves; call ``reset()`` after a tap."""

    def __init__(self, frames: int = 2) -> None:
        self.frames = frames
        self._key: Optional[Tuple[Pos, str]] = None
        self.count = 0

    def feed(self, res: Analysis) -> Optional[Candidate]:
        move = (res.decision.move if res.failure is None and res.decision is not None
                else None)
        if move is None:
            self.reset()
            return None
        key = (move.pos, move.color)
        self.count = self.count + 1 if key == self._key else 1
        self._key = key
        return move if self.count >= self.frames else None

    def reset(self) -> None:
        self._key, self.count = None, 0
