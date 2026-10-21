"""Phase 6 part of bot.py: level start / end and the screens outside gameplay.

``LevelMixin`` is mixed into ``bot.Bot`` (uses its log, clock, tapper, geometry, analyzer,
session and flow). A level starts on its first gameplay frame and ends on a confirmed
win / fail / revive screen (or an inferred win: almost no sand left, then another screen);
``flow.ScreenFlow`` picks the only button allowed on each non-gameplay frame, and every such
tap is logged with its raw + annotated frame (event ``screen_tap``).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from flow import draw_screen_tap

RESULTS = {"level_complete": "win", "fail": "fail", "revive": "fail"}


class Stop(Exception):
    def __init__(self, result: str, reason: str) -> None:
        super().__init__(reason)
        self.result, self.reason = result, reason


class LevelMixin:
    def _start_level(self, now: float) -> None:
        self._reset_level(analyzer=False)   # reset when the last level ended
        self.in_level, self.level_start = True, now
        self.level_no += 1
        self.level_taps, self.level_last, self.level_t0 = [], None, self.log.t()
        self.level_first, self.level_sig = None, None
        self.speed_taps, self.speed_tap_t, self.speed_gave_up = 0, -1e9, False
        lives = "?" if self.lives is None else self.lives
        self.log.log(f"LEVEL {self.level_no} started (lives {lives})")
        self.log.event("level_start", level=self.level_no, frame=self.frame_no, lives=self.lives)

    def _level_left(self, res: Any, frame: np.ndarray) -> None:
        """Gameplay left for a confirmed other screen: win / fail, or not over yet."""
        result, how = RESULTS.get(res.screen), f"screen {res.screen}"
        sand = self.last_sand
        start = (self.level_first or {}).get("sand_total") or 0.0
        # an inferred win needs evidence (run 20261021-112241: a phantom "level" on the finished
        # board, 0 taps, sand 0 % from its first frame, "won" at the next screen change): this
        # level's own taps took the sand from >= infer_win_start_sand to ~0
        if (result is None and sand is not None and sand < self.bc.win_sand_max
                and self.level_taps and start >= self.bc.infer_win_start_sand):
            result, how = "win", f"inferred: sand left {sand:.1%}, then screen {res.screen}"
        if result is None:
            if not self.left_unclear:
                self.left_unclear = True
                png = self.log.frame(f"{self.frame_no:05d}-screen-{res.screen}", frame)
                self.log.log(f"level {self.level_no} left for screen {res.screen} with sand "
                             f"{self.last_sand} - not counted as over yet (frame {png})")
            return
        png = self.log.frame(f"{self.frame_no:05d}-screen-{res.screen}", frame)
        self.in_level = False
        self.log.log(f"LEVEL {self.level_no}: {result.upper()} ({how})")
        self.log.event("level_end", level=self.level_no, result=result, how=how, png=png,
                       taps=self.taps)
        self._record_level(result, how)   # before the reset: it keeps the level's plans
        if hasattr(self.analyzer, "reset_level"):
            self.analyzer.reset_level()
        stop = self.session.record(result)
        if stop is not None:
            raise Stop(result, stop)

    def _record_level(self, result: str, how: str) -> None:
        """Phase 7: every level -> <debug>/levels.jsonl: result, every tap with the decision it
        came from (state, candidates, scores, lookahead) and the last gameplay read."""
        an = self.analyzer
        planner = getattr(an, "planner", None)
        self.log.level(level=self.level_no, result=result, how=how, signature=self.level_sig,
                       duration_s=round(self.log.t() - self.level_t0, 1),
                       taps=len(self.level_taps), tap_log=self.level_taps,
                       start=self.level_first, end=self.level_last,
                       plans=planner.log if planner is not None else [],
                       searches=getattr(getattr(an, "decider", None), "log", None) or [],
                       chests=list(getattr(an, "chest_log", [])),
                       speed_x2=getattr(getattr(an, "speed", None), "state", None),
                       revealed=dict(getattr(an, "revealed", {}) or {}))

    def _non_gameplay(self, res: Any, frame: np.ndarray, now: float) -> None:
        from hud import read_lives
        from screens import ScreenResult

        self.non_game += 1
        self.gate.reset()
        self.log.event("screen", frame=self.frame_no, screen=res.screen, count=self.non_game)
        if self.in_level:
            if self.non_game < self.bc.screen_confirm_frames:
                return
            self._level_left(res, frame)
        lives = None
        if hasattr(self.analyzer, "palette"):
            lives = read_lives(frame, self.analyzer.palette, self.analyzer.reader, self.cfg.flow)
        if lives is not None and lives != self.lives:
            self.log.log(f"lives: {lives}")
            self.log.event("lives", lives=lives, frame=self.frame_no)
            self.lives = lives
        act = self.flow.feed(res.screen_result or ScreenResult(res.screen), frame, lives, now)
        if act.kind == "stop":
            png = self.log.frame(f"{self.frame_no:05d}-stop-{res.screen}", frame)
            raise Stop("stopped", f"{act.reason} (frame {png})")
        if act.kind == "tap":
            self._screen_tap(act, res, frame, lives)
        elif act.reason != self.last_screen_reason:
            png = self.log.frame(f"{self.frame_no:05d}-screen-{res.screen}-wait", frame)
            self.log.log(f"screen {res.screen}: {act.reason}")
            self.log.event("screen_wait", frame=self.frame_no, screen=res.screen,
                           reason=act.reason, png=png)
            self.screen_frame_t = now
        elif (res.screen in ("other", "ad")
              and now - getattr(self, "screen_frame_t", -1e9) >= self.cfg.flow.unknown_frame_s):
            self.screen_frame_t = now   # unknown / ad screens: a frame every unknown_frame_s
            self.log.frame(f"{self.frame_no:05d}-screen-{res.screen}-wait", frame)
        self.last_screen_reason = act.reason

    def _finished_board(self, res: Any) -> bool:
        """Run 20261021-000105 (first L187 win): between the win screen / ads the game showed
        the finished level's board again - empty grid, 0/5, no sand - and each such frame
        started a "level" that was won at once (3 wins counted for one). A level starts only on
        a board with something to play: a bucket, lock or unknown cell, or sand. Such a frame
        is skipped and the Analyzer forgets it (its grid lock, its empty capacities).
        Run 20261021-112241: the finished L189 board fading back in after an ad read its empty
        cells as unknown (LAB 80/-1/-13) and started a phantom level - a level always starts
        with sand in the picture, so a board without sand never starts one."""
        st = res.state
        if st is None:
            return False
        if sum(st.sand.values()) >= self.bc.win_sand_max:
            return False
        if self.last_screen_reason != "finished board":
            self.log.log("gameplay frame without sand (the finished level) - not a new level")
            self.log.event("finished_board", frame=self.frame_no)
            self.last_screen_reason = "finished board"
        if hasattr(self.analyzer, "reset_level"):
            self.analyzer.reset_level()
        return True

    def _speed_up(self, res: Any, frame: np.ndarray, now: float) -> bool:
        """User (2026-10-20): the ▶▶2x belt speed button is free and resets every level. While
        it reads off (hud.SpeedTracker, confirmed), tap it - only it - at most
        ``bot.speed_max_taps`` times per level, ``speed_retap_s`` apart. True = this frame was
        spent on it (no bucket move)."""
        from actions import OutOfBounds

        bc = self.bc
        if not bc.speed_x2 or getattr(res, "speed_x2", None) is not False:
            return False
        if now - self.speed_tap_t < bc.speed_retap_s:
            return True                    # just tapped: wait for the button to turn green
        if self.speed_taps >= bc.speed_max_taps:
            if not self.speed_gave_up:
                self.speed_gave_up = True
                png = self.log.frame(f"{self.frame_no:05d}-speed-still-off", frame)
                self.log.log(f"belt speed 2x still off after {self.speed_taps} tap(s) - playing "
                             f"at 1x (frame {png})")
            return False
        geo = self.geometry(force=True)
        if tuple(geo.frame_size) != (frame.shape[1], frame.shape[0]):
            return True
        z = bc.speed_zone
        rel = ((z[0] + z[2]) / 2, (z[1] + z[3]) / 2)
        try:
            tr = self.tapper.tap(*rel)
        except OutOfBounds as e:
            self.log.log(f"speed tap refused: {e}")
            self.speed_taps = bc.speed_max_taps
            return False
        self.speed_taps += 1
        self.speed_tap_t = now
        self.log.log(f"SPEED TAP ▶▶2x ({self.speed_taps}/{bc.speed_max_taps}): belt speed reads "
                     f"1x{' [dry-run]' if self.dry_run else ''}")
        self.log.event("speed_tap", n=self.speed_taps, frame=self.frame_no,
                       rel=[round(rel[0], 4), round(rel[1], 4)], executed=tr.executed,
                       frame_png=self.log.frame(f"{self.frame_no:05d}-speed-tap", frame))
        return True

    def _screen_tap(self, act: Any, res: Any, frame: np.ndarray, lives: Optional[int]) -> None:
        """One confirmed tap on a non-gameplay screen (flow.py chose the button)."""
        from actions import OutOfBounds

        geo = self.geometry(force=True)   # geometry check right before tapping
        if tuple(geo.frame_size) != (frame.shape[1], frame.shape[0]):
            self.log.log(f"geometry changed since the frame - no {act.button} tap")
            return
        raw = self.log.frame(f"{self.frame_no:05d}-screen-tap-{act.button}", frame)
        ann = self.log.frame(f"{self.frame_no:05d}-screen-tap-{act.button}-annotated",
                             draw_screen_tap(frame, act, res.screen_result))
        try:
            tr = self.tapper.tap(*act.rel)
        except OutOfBounds as e:
            self.log.log(f"screen tap refused: {e}")
            self.log.event("log", msg=f"screen tap refused: {e}")
            return
        self.screen_taps += 1
        self.flow.tapped(act, self.clock())
        self.log.log(f"SCREEN TAP #{self.screen_taps} {act.screen}: {act.button} ({act.reason})"
                     f"{' [dry-run]' if self.dry_run else ''}")
        self.log.event("screen_tap", n=self.screen_taps, frame=self.frame_no, screen=act.screen,
                       button=act.button, score=round(act.score, 3),
                       rel=[round(act.rel[0], 4), round(act.rel[1], 4)],
                       point=[round(tr.point[0], 1), round(tr.point[1], 1)],
                       executed=tr.executed, lives=lives, frame_png=raw, annotated_png=ann)
