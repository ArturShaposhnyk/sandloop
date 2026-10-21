"""Sand Loop bot: plays levels by itself with real taps (phase 5 gameplay, phase 6 screens).

    python -m bot --dry-run            # full loop, logs every decision, never clicks
    python -m bot --max-taps 5         # real taps, stop after 5 (first live test)
    python -m bot                      # real taps until one level ends
    python -m bot --levels 10          # 10 levels in a row (win -> NEXT -> PLAY, fail -> retry)

Loop: capture -> screen type -> (gameplay) analyse -> 2-frame confirmed decision -> tap(s)
via actions.Tapper. Safety (never remove): cursor in a screen corner = stop (checked every
loop and before every tap), ctrl+alt+p pause / ctrl+alt+q quit (config.yaml safety:), taps-per-
second limit, taps only inside the detected iPhone screen rect, geometry re-checked right before
tapping (window moved / resized since the analysed frame -> no tap), --dry-run.

After a tap: the decision gate restarts, the tapped cells are remembered (no re-tap while the
bucket flies; towers keep their stages for the belt), and only the grid + k/5 counter must
settle (sand never stops moving). Tempo (tempo.py): up to bot.burst_max buckets back to back
from one confirmed decision; colours being covered get a bonus. Non-gameplay screens (win,
menu, revive, fail, ads, popups): flow.py decides (only NEXT / PLAY / Quit / Try Again / ✕),
every such tap is logged with its frame. A level ends on its win / fail / revive screen; the
session stops after --levels N, at 0 lives or at a config.yaml session: limit. Every run is
logged to debug/runs/<stamp>/ (runlog.py); post-mortem: ``python -m tools.analyze_run``.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cells import Kind
from config import Config, load_config
from flow import ScreenFlow, Session
from levels import LevelMixin, Stop
from runlog import RunLog, analysis_record, prune_crops
from tempo import ExposureTrend, ReleaseLedger, burst, prioritize

def zone_change(a: np.ndarray, b: np.ndarray, zones: Sequence[Sequence[float]]) -> float:
    """Largest frame change (capture.frame_change) over the given relative zones."""
    from capture import frame_change
    from zones import crop_rel

    if a.shape != b.shape:
        return 1.0
    return max((frame_change(crop_rel(a, z), crop_rel(b, z)) for z in zones), default=0.0)


class Bot(LevelMixin):
    """``grab()`` -> iPhone screen frame (raises capture errors), ``geometry(force)`` ->
    capture.ScreenGeometry, ``tapper.tap(rx, ry)``, ``controls`` has ``paused`` / ``quit``
    events (safety.Hotkeys), ``failsafe.check()`` raises safety.FailsafeTriggered."""

    def __init__(self, cfg: Config, analyzer: Any, tapper: Any, grab: Callable[[], np.ndarray],
                 geometry: Callable[..., Any], log: RunLog, controls: Any, failsafe: Any,
                 dry_run: bool = False, max_taps: Optional[int] = None, levels: int = 1,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 diagnose: Optional[Callable[[], str]] = None) -> None:
        from analyzer import DecisionGate

        self.cfg, self.bc = cfg, cfg.bot
        self.crop_t = -1e9                    # last picture + belt crop (replay data)
        self.analyzer, self.tapper, self.grab, self.geometry = analyzer, tapper, grab, geometry
        self.log, self.controls, self.failsafe = log, controls, failsafe
        self.dry_run, self.max_taps = dry_run, max_taps
        self.clock, self.sleep = clock, sleep
        self.diagnose = diagnose              # window lost: what the window list shows
        self.gate = DecisionGate(cfg.strategy.confirm_frames)
        self.flow = ScreenFlow(cfg.flow, dry_run)
        self.session = Session(levels, cfg.session)
        self.zones = [getattr(cfg.vision, z) for z in self.bc.settle_zones]
        self.frame_no = 0
        self.taps = 0
        self.screen_taps = 0
        self.level_no = 0
        self.level_taps: List[Dict[str, Any]] = []      # this level's tap events (levels.jsonl)
        self.level_last: Optional[Dict[str, Any]] = None   # ... its last gameplay analysis
        self.level_t0 = 0.0
        self.level_first: Optional[Dict[str, Any]] = None   # its first gameplay analysis
        self.level_sig: Optional[str] = None                # board signature (levelplan)
        self.in_level = False
        self.level_start: Optional[float] = None
        self.left_unclear = False             # left a level to a screen that is no result
        self.lives: Optional[int] = None
        self.last_screen_reason: Optional[str] = None
        self.prev_frame: Optional[np.ndarray] = None
        self.non_game = 0
        self.last_reason: Optional[str] = None
        self.lost_since: Optional[float] = None
        self.last_frame: Optional[np.ndarray] = None
        self.effect_retry = False             # the tap effect check waits one more frame
        self.was_paused = False
        self._reset_level()

    def _reset_level(self, analyzer: bool = True) -> None:
        """Forget everything that belongs to one level. ``analyzer``: also the Analyzer's level
        memory (grid geometry, ...) - not when the level's first frame is already read."""
        if analyzer and hasattr(self.analyzer, "reset_level"):
            self.analyzer.reset_level()
        self.gate.reset()
        self.trend = ExposureTrend(self.bc.cover_window_s, self.bc.cover_min_drop)
        self.ledger = ReleaseLedger(self.bc.retap_guard_s)
        self.prev_game: Any = None            # last gameplay Analysis with a state
        self.settle_from: Optional[float] = None
        self.pending: Optional[Tuple[List[Tuple[Tuple[int, int], str]], Optional[int]]] = None
        self.no_effect = 0
        self.dead_taps: Dict[Tuple[int, int], Any] = {}   # cell -> (board, t) of a tap that did nothing
        self.last_sand: Optional[float] = None
        self.disagree_since: Dict[Tuple[int, int], float] = {}   # cell -> look != path rule since
        self.disagree_reported = -1e9

    # -- main loop --------------------------------------------------------------------------

    def run(self) -> str:
        from safety import FailsafeTriggered

        start = self.clock()
        result, reason = "stopped", "?"
        try:
            while True:
                self._checks(start)
                frame = self._capture()
                if frame is None or self._settling(frame):
                    continue
                self._analyse(frame)
        except Stop as s:
            result, reason = s.result, s.reason
        except FailsafeTriggered as e:
            result, reason = "stopped", f"failsafe: {e}"
        except KeyboardInterrupt:
            result, reason = "stopped", "Ctrl+C"
        if self.in_level:
            self._record_level("unfinished", f"{result}: {reason}")
        summary = self.session.summary()
        self.log.log(f"RESULT: {result} - {reason} ({self.taps} taps, {self.screen_taps} screen "
                     f"taps, levels {summary['levels']}: {summary['wins']} won, "
                     f"{summary['fails']} lost)")
        self.log.result(result, reason, taps=self.taps, frames=self.frame_no,
                        dry_run=self.dry_run, last_sand_total=self.last_sand,
                        released=list(self.ledger.describe()), screen_taps=self.screen_taps,
                        lives=self.lives, **summary)
        return result

    def _checks(self, start: float) -> None:
        if self.controls.quit.is_set():
            raise Stop("stopped", "quit hotkey")
        self.failsafe.check()
        now = self.clock()
        if now - start > self.cfg.session.max_total_minutes * 60:
            raise Stop("stopped", f"session.max_total_minutes "
                                  f"{self.cfg.session.max_total_minutes:.0f} reached")
        if self.in_level and now - self.level_start > self.bc.max_run_s:
            raise Stop("stopped", f"level {self.level_no}: max_run_s {self.bc.max_run_s:.0f}s "
                                  "reached")
        if self.max_taps is not None and self.taps >= self.max_taps and self.pending is None:
            raise Stop("stopped", f"--max-taps {self.max_taps} reached")
        while self.controls.paused.is_set() and not self.controls.quit.is_set():
            if not self.was_paused:
                self.log.log("paused (hotkey again to resume)")
                self.log.event("log", msg="paused")
                self.was_paused = True
            self.gate.reset()
            self.prev_game = None
            self.failsafe.check()
            self.sleep(0.1)
        if self.was_paused:
            self.log.log("resumed")
            self.was_paused = False

    def _capture(self) -> Optional[np.ndarray]:
        from capture import CaptureError, WindowNotFound

        self.sleep(self.bc.loop_interval_s)
        try:
            frame = self.grab()
        except (WindowNotFound, CaptureError) as e:
            now = self.clock()
            if self.lost_since is None:
                self.lost_since = now
                self.log.log(f"window / geometry problem: {e} - waiting up to "
                             f"{self.bc.window_lost_max_s:.0f}s for it")
                self.log.event("log", msg=f"window lost: {e}")
                self._lost_report()
            if now - self.lost_since > self.bc.window_lost_max_s:
                raise Stop("stopped", f"window lost for {self.bc.window_lost_max_s:.0f}s: {e}")
            self.gate.reset()
            self.prev_game = None
            self.sleep(0.5)
            return None
        if self.lost_since is not None:
            self.log.log(f"window back after {self.clock() - self.lost_since:.1f}s")
            self.log.event("log", msg="window back")
            self.lost_since = None
        self.frame_no += 1
        self.last_frame = frame
        return frame

    def _lost_report(self) -> None:
        """Window lost: save the last frame before it and the window list (run 161536:
        Mirroring vanished at 76 s with no frame of what happened)."""
        if self.last_frame is not None:
            path = self.log.frame(f"{self.frame_no:05d}-before-window-lost", self.last_frame)
            self.log.log(f"last frame before the window was lost: {path}")
        if self.diagnose is not None:
            try:
                info = self.diagnose()
            except Exception as e:   # noqa: BLE001 - diagnostics only
                info = f"window list failed: {e!r}"
            self.log.log(f"window list: {info}")
            self.log.event("log", msg=f"window list: {info}")

    def _settling(self, frame: np.ndarray) -> bool:
        """After a tap: True (skip this frame) until the grid / counter calmed down."""
        prev, self.prev_frame = self.prev_frame, frame
        if self.settle_from is None:
            return False
        age = self.clock() - self.settle_from
        if age < self.bc.settle_min_s:
            return True
        if (age < self.bc.settle_max_s and prev is not None
                and zone_change(prev, frame, self.zones) >= self.bc.settle_threshold):
            return True
        self.settle_from = None
        return False

    # -- one analysed frame -----------------------------------------------------------------

    def _analyse(self, frame: np.ndarray) -> None:
        now = self.clock()
        res = self.analyzer.analyze(frame, log=self.log.log, now=now,
                                    adjust=lambda st: self._adjust(st, now))
        if res.screen != "gameplay":
            self._non_gameplay(res, frame, now)
            return
        if not self.in_level and self._finished_board(res):
            return                            # the last level's empty board, not a new one
        self.non_game = 0
        self.flow.in_gameplay()
        self.last_screen_reason = None
        if not self.in_level:
            self._start_level(now)
        if self._speed_up(res, frame, now):   # level start: ▶▶2x first (levels.py)
            self.gate.reset()
            return
        self.left_unclear = False
        st = res.state
        if st is not None:
            self.trend.update(now, st.exposed)
            self.last_sand = round(sum(st.sand.values()), 4)
            if res.decision is not None:
                covering = self.trend.covering(b.color for b in st.conveyor if b.color)
                res.decision = prioritize(res.decision, covering, self.bc.weight_covering)
            self._verify(st)
        confirmed = self.gate.feed(res)
        rec = self.log.event("analysis", frame=self.frame_no, gate=self.gate.count,
                             confirmed=confirmed is not None, **analysis_record(res))
        if st is not None:
            self.level_last = rec
            if self.level_first is None and res.view is not None:
                from levelplan import level_signature

                self.level_first, self.level_sig = rec, level_signature(res.view.labels())
        self.log.belt(self.frame_no, res)
        if st is not None and self.bc.crop_every_s and now - self.crop_t >= self.bc.crop_every_s:
            self.crop_t = now        # replay data (tools/replay.py)
            self.log.crop(self.frame_no, frame, self.bc.crop_rows, self.bc.crops_max_mb)
        self._report(res, frame, confirmed is not None, rec)
        self._watch_disagreements(res, frame, now)
        if confirmed is not None:
            self._play(res, frame, confirmed, now)
        if st is not None and res.failure is None:
            self.prev_game = res

    def _report(self, res: Any, frame: np.ndarray, confirmed: bool, rec: Dict) -> None:
        if res.failure:
            reason = f"read failed ({res.failure}) - wait"
        elif res.decision is None:
            reason = "no decision"
        elif res.decision.move is not None and not confirmed:
            return   # pending: printed once confirmed
        else:
            reason = res.decision.reason
        if reason != self.last_reason:
            self.log.log(reason + (" CONFIRMED" if confirmed else ""))
            if not confirmed:
                self.log.frame(f"{self.frame_no:05d}-wait", frame)
            self.last_reason = reason

    def _watch_disagreements(self, res: Any, frame: np.ndarray, now: float) -> None:
        """A cell whose look and path rule disagree is never tapped. When that has lasted
        ``disagree_report_s`` and the bot has no other move, say so and save the frame (again
        every ``disagree_report_s``) instead of waiting silently (live run 20261019-162703)."""
        view = res.view
        if res.failure or view is None:
            return
        self.disagree_since = {p: self.disagree_since.get(p, now)
                               for p in map(tuple, view.disagreements)}
        old = sorted(p for p, t in self.disagree_since.items()
                     if now - t >= self.bc.disagree_report_s)
        waiting = res.decision is not None and res.decision.move is None
        if not (waiting and old) or now - self.disagree_reported < self.bc.disagree_report_s:
            return
        self.disagree_reported = now
        cells = [f"r{r}c{c} {view.grid[r][c].label()}" for r, c in old]
        png = self.log.frame(f"{self.frame_no:05d}-stuck-disagree", frame)
        self.log.log(f"STUCK: no move, and {', '.join(cells)} look different from what the path "
                     f"rule says for >= {self.bc.disagree_report_s:.0f}s (misread grid?) - frame "
                     f"saved: {png}")
        self.log.event("stuck", frame=self.frame_no, cells=cells, png=png)

    def _adjust(self, st: Any, now: float) -> None:
        """The state before the decision: released towers (ledger) and no tap on a cell whose
        tap had no effect until the board changes (or ``bot.dead_tap_s`` passed: then one more
        try, and the no-effect stop applies if that fails too)."""
        self.ledger.update(st, now)
        if not self.dead_taps:
            return
        from sim import grid_signature

        board = grid_signature(st.grid)
        for pos, (sig, t) in list(self.dead_taps.items()):
            if sig != board or now - t >= self.bc.dead_tap_s:
                del self.dead_taps[pos]
            elif pos in st.tappable:
                st.tappable = [p for p in st.tappable if p != pos]

    def _verify(self, st: Any) -> None:
        """First settled gameplay frame after taps: did the tapped cells change?"""
        if self.pending is None or self.dry_run:
            self.pending = None
            return
        tapped, k_before = self.pending
        self.pending = None
        changed = 0
        for (r, c), color in tapped:
            if r >= st.rows or c >= st.cols:
                return   # lattice changed: can't tell
            cell = st.cell((r, c))
            if cell is None or cell.kind is not Kind.BUCKET or cell.color != color:
                changed += 1
        k_up = (st.conveyor_k is not None and k_before is not None and st.conveyor_k > k_before)
        ok = changed > 0 or k_up
        if not ok and not self.effect_retry:   # the frame may be stale (run 161536, TAP #1)
            self.effect_retry = True
            self.pending = (tapped, k_before)
            self.log.log("tap effect not visible yet - checking the next frame")
            return
        self.effect_retry = False
        self.log.event("effect", ok=ok, changed=changed, taps=len(tapped), k_before=k_before,
                       k_after=st.conveyor_k)
        if ok:
            self.no_effect = 0
            return
        self.no_effect += 1
        from sim import grid_signature

        for pos, _ in tapped:   # not again on this board (run 20261021-143729: r6c2 twice)
            self.dead_taps[tuple(pos)] = (grid_signature(st.grid), self.clock())
        self.log.log(f"warning: tap had no visible effect ({self.no_effect}/"
                     f"{self.bc.max_no_effect_taps})")
        if self.no_effect >= self.bc.max_no_effect_taps:
            raise Stop("stopped", f"{self.no_effect} taps in a row had no effect")

    # -- tapping ----------------------------------------------------------------------------

    def _play(self, res: Any, frame: np.ndarray, move: Any, now: float) -> None:
        from actions import OutOfBounds
        from strategy import draw_decision

        st = res.state
        self.gate.reset()
        if self.ledger.guarded(move.pos, now):
            self.log.log(f"r{move.pos[0]}c{move.pos[1]} was just tapped - not again yet")
            return
        if self.dead_taps.get(tuple(move.pos)) is not None:
            from sim import grid_signature

            sig, t = self.dead_taps[tuple(move.pos)]
            if sig == grid_signature(st.grid) and now - t < self.bc.dead_tap_s:
                self.log.log(f"r{move.pos[0]}c{move.pos[1]}: its tap had no effect on this "
                             "board - not again until the board changes")
                return
        both = set(st.tappable) & (set(self.prev_game.state.tappable)
                                   if self.prev_game is not None else set(st.tappable))
        both = {p for p in both if not self.ledger.guarded(p, now)}
        covering = self.trend.covering(b.color for b in st.conveyor if b.color)
        moves = ([(move, f"confirmed, level plan step {move.level_plan}")]
                 + [(f, f"burst {k + 2}: level plan step {f.level_plan}")
                    for k, f in enumerate(itertools.takewhile(lambda f: f.pos in both,
                                                              move.follow))]
                 if move.level_plan
                 else burst(st, move, both, self.cfg.strategy, self.bc, covering))
        if self.max_taps is not None:
            if self.taps >= self.max_taps:
                return   # --max-taps reached: the loop stops once the last tap is checked
            moves = moves[:self.max_taps - self.taps]
        geo = self.geometry(force=True)   # geometry check right before tapping
        if tuple(geo.frame_size) != (frame.shape[1], frame.shape[0]):
            self.log.log(f"geometry changed since the frame ({geo.frame_size} vs "
                         f"{frame.shape[1]}x{frame.shape[0]}) - no tap")
            self.log.event("log", msg="geometry changed - no tap")
            return
        raw = self.log.frame(f"{self.frame_no:05d}-tap", frame)
        ann = self.log.frame(f"{self.frame_no:05d}-tap-annotated",
                             draw_decision(frame, res.view, res.decision))
        tapped: List[Tuple[Tuple[int, int], str]] = []
        last = None
        for i, (cand, why) in enumerate(moves):
            if i > 0:
                self.failsafe.check()
                if self.controls.quit.is_set() or self.controls.paused.is_set():
                    break
                wait = self.bc.burst_gap_s - (self.clock() - last)
                if wait > 0:
                    self.sleep(wait)
            rel = res.view.cell_center_rel(*cand.pos)
            try:
                tr = self.tapper.tap(*rel)
            except OutOfBounds as e:
                self.log.log(f"tap refused: {e}")
                self.log.event("log", msg=f"tap refused: {e}")
                break
            last = self.clock()
            self.taps += 1
            cell = st.cell(cand.pos)
            self.ledger.add(last, cand.pos, cand.color, cell.body if cell is not None else None)
            tapped.append((cand.pos, cand.color))
            planner = getattr(self.analyzer, "planner", None)
            if planner is not None:
                planner.tapped(cand.pos)   # whole-level sequences for the level memory
            r, c = cand.pos
            self.log.log(f"TAP #{self.taps} r{r}c{c} {cand.color} ({why})"
                         f"{' [dry-run]' if self.dry_run else ''}")
            evt = self.log.event("tap", n=self.taps, frame=self.frame_no, burst=i + 1,
                                 pos=[r, c], color=cand.color, stages=list(cand.stages),
                                 score=round(cand.score, 4), avail=round(cand.avail, 4),
                                 exposed=round(cand.exposed, 4),
                                 k_before=None if st.conveyor_k is None else st.conveyor_k + i,
                                 free_before=None if st.free_slots is None else st.free_slots - i,
                                 why=why, rel=[round(rel[0], 4), round(rel[1], 4)],
                                 point=[round(tr.point[0], 1), round(tr.point[1], 1)],
                                 executed=tr.executed, frame_png=raw, annotated_png=ann,
                                 gamble=cand.gamble, lost=cand.lost,
                                 lookahead=cand.outlook.describe() if cand.outlook else None,
                                 alternatives=sum(1 for x in res.decision.candidates
                                                  if x.rejected is None and x.pos != cand.pos))
            self.level_taps.append({**evt, "decision": self.level_last} if i == 0 else evt)
        if tapped:
            self.pending = (tapped, st.conveyor_k)
            self.effect_retry = False
            self.settle_from = self.clock()
            self.prev_game = None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bot", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="never click (logs the taps)")
    ap.add_argument("--max-taps", type=int, default=None, help="stop after this many taps")
    ap.add_argument("--levels", type=int, default=1,
                    help="levels to play in a row (default 1; capped by session.max_levels)")
    ap.add_argument("--countdown", type=float, default=3.0,
                    help="seconds before the first capture (default 3)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    if args.max_taps is not None and args.max_taps < 1:
        ap.error("--max-taps must be >= 1")
    if args.levels < 1:
        ap.error("--levels must be >= 1")

    import permissions
    from actions import Tapper
    from capture import Grabber, WindowTracker, describe_windows
    from safety import Failsafe, Hotkeys
    from analyzer import Analyzer

    cfg = load_config(args.config)
    permissions.require(screen=True, accessibility=not args.dry_run)
    grabber = Grabber(cfg.screen.capture)
    tracker = WindowTracker(cfg, grabber)
    log = RunLog(cfg.paths.debug, cfg, vars(args), cfg.bot.max_saved_frames)
    if cfg.bot.crops_max_mb:
        prune_crops(cfg.paths.debug, cfg.bot.crops_max_mb, keep=log.dir, log=log.log)
    hotkeys = Hotkeys(cfg.safety.hotkey_pause, cfg.safety.hotkey_quit).start()
    failsafe = Failsafe(cfg.safety.failsafe_corner_pt)
    log.log(f"run log: {log.dir}")
    log.log(f"{'DRY RUN (no clicks)' if args.dry_run else 'LIVE: real taps'}; "
            f"pause {cfg.safety.hotkey_pause}, quit {cfg.safety.hotkey_quit}, "
            f"cursor in a screen corner = stop")
    try:
        analyzer = Analyzer(cfg)
        tapper = Tapper(cfg, tracker, dry_run=args.dry_run, failsafe=failsafe, log=log.log)
        bot = Bot(cfg, analyzer, tapper, tracker.grab, tracker.get, log, hotkeys, failsafe,
                  dry_run=args.dry_run, max_taps=args.max_taps, levels=args.levels,
                  diagnose=lambda: describe_windows(cfg.window.owner_name))
        end = time.monotonic() + args.countdown
        while time.monotonic() < end and not hotkeys.quit.is_set():
            failsafe.check()
            time.sleep(0.1)
        if not hotkeys.quit.is_set():
            tapper.prepare()   # iPhone Mirroring frontmost before the first tap
        result = bot.run()
    except Exception as e:   # noqa: BLE001 - record any crash in the run log, then re-raise
        if type(e).__name__ != "FailsafeTriggered":
            log.result("stopped", f"crash: {e!r}")
            raise
        log.result("stopped", f"failsafe: {e}")
        result = "stopped"
    finally:
        hotkeys.quit.set()
        log.close()
        grabber.close()
    print(f"post-mortem: python -m tools.analyze_run {log.dir}")
    return 0 if result == "win" else 1


if __name__ == "__main__":
    sys.exit(main())
