"""Frame -> decision, shared by bot.py, the overlay and ``python -m strategy``: ``Analyzer``
reads a frame (screen type, grid, belt) and runs ``strategy.choose``; ``DecisionGate`` confirms a
move over consecutive frames. Split out of strategy.py; never taps."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

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
        self.icons = IconMatcher.load(Path(cfg.paths.dataset) / "templates" / "icons")
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
        from hints import CellMemory, LatticeLock, RecentUnlocks

        cfg = self.cfg
        self.memory = CellMemory()   # last clean cell readings, for cells under hint arrows
        self.unlocks = RecentUnlocks(cfg.vision.unlock_trust_s)   # key locks that just opened
        self.lattices: Optional[LatticeLock] = (
            LatticeLock(cfg.vision.lattice_switch_frames) if track else None)
        self.chests: Dict[Pos, int] = {}       # chests in the last read (cell -> N)
        self.scatter_until = -1.0              # a chest opened: no decisions until then

    def analyze(self, img: "np.ndarray", log=print, now: Optional[float] = None,
                diff: Optional[float] = None, adjust=None) -> Analysis:
        """``now``: frame time in seconds (default: ``self.clock()``) for the belt stall clock.
        ``diff``: frame-to-frame change at capture, logged with failed reads. ``adjust(state)``
        may amend the state before the decision (bot.py: what it knows of released towers)."""
        from vision import VisionError, read_gameplay

        now = self.clock() if now is None else now
        found = self.classifier.classify(img)
        screen = found.screen
        if screen != "gameplay":
            if self.lattices is not None:
                self.lattices.ref = None   # another level may follow
            return Analysis(screen, reads=self.reads, fails=self.fails, screen_result=found)
        self.reads += 1
        try:
            gv = read_gameplay(img, self.palette, self.cfg.vision, self.reader, log=log,
                               icons=self.icons, hints=self.hints, memory=self.memory,
                               unlocks=self.unlocks, now=now, lattices=self.lattices)
        except VisionError as e:
            res = Analysis(screen, error=str(e), failure=f"grid: {e}")
        else:
            self._save_unknowns(img, gv, log)
            st = self.cfg.strategy
            state = LevelState.from_view(gv, st.conveyor_slots, st.unknown_fill_need)
            state.belt_stalled = self.belt.update(state.conveyor, now)
            if adjust is not None:
                adjust(state)
            res = Analysis(screen, gv, state, choose(state, st))
            if self.lattices is not None:
                self._chest_scatter(gv, res, now, log)
            if gv.counter is None:
                res.failure = "counter: k/5 not read"
        if res.failure:
            self.fails += 1
            self._save_failure(img, res.failure, diff, log)
        res.reads, res.fails = self.reads, self.fails
        return res

    def _chest_scatter(self, gv: "GameView", res: Analysis, now: float, log) -> None:
        """A chest that was in the last read and is gone has opened: its buckets scatter over
        the grid (GAME_RULES.md, L186). Forget the old cell readings (hint-arrow memory) and
        decide nothing for ``chest_settle_s``: every frame is a full re-read anyway."""
        chests = {(c.row, c.col): c.counter or 1 for row in gv.grid for c in row
                  if c is not None and c.kind is Kind.LOCK and c.lock == "chest"}
        opened = {p: n for p, n in self.chests.items() if p not in chests}
        self.chests = chests
        if opened:
            from hints import CellMemory

            self.memory = CellMemory()
            self.scatter_until = now + self.cfg.strategy.chest_settle_s
            log("chest " + ", ".join(f"r{r}c{c} ({n})" for (r, c), n in opened.items())
                + " opened: buckets scatter - re-reading the whole grid")
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
