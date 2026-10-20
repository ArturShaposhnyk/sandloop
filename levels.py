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
        lives = "?" if self.lives is None else self.lives
        self.log.log(f"LEVEL {self.level_no} started (lives {lives})")
        self.log.event("level_start", level=self.level_no, frame=self.frame_no, lives=self.lives)

    def _level_left(self, res: Any, frame: np.ndarray) -> None:
        """Gameplay left for a confirmed other screen: win / fail, or not over yet."""
        result, how = RESULTS.get(res.screen), f"screen {res.screen}"
        sand = self.last_sand
        if result is None and sand is not None and sand < self.bc.win_sand_max:
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
        if hasattr(self.analyzer, "reset_level"):
            self.analyzer.reset_level()
        self.log.log(f"LEVEL {self.level_no}: {result.upper()} ({how})")
        self.log.event("level_end", level=self.level_no, result=result, how=how, png=png,
                       taps=self.taps)
        stop = self.session.record(result)
        if stop is not None:
            raise Stop(result, stop)

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
        self.last_screen_reason = act.reason

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
