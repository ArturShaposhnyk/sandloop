"""Phase 6: what the bot does on screens outside gameplay, and the session limits.

GAME_RULES.md "Bot behaviour outside gameplay" as data (``plan``):

    level_complete  NEXT                         (never Watch Ad / x2)
    menu            red ✕ of a popup, else PLAY  (offers, passes, events: only ✕)
    revive          Quit, else the popup's red ✕ (never Keep Playing: coins)
    fail            Try Again
    ad              "Take a break!" -> wait; else the generic ✕ (adclose.py) once it shows,
                    else an ad's disc button (▶| skip / small ✕, adclose.find_skip); never Download / install
    other           wait (maybe an ad: ✕ / skip too, within flow.ad_context_s of an ad or
                    flow.ad_after_tap_s of a screen tap - run 171133: ads without any anchor
                    after Try Again)

Taps only at a matched anchor's centre or a found ✕, never inside a forbidden anchor
(screens.FORBIDDEN), only after ``flow.confirm_frames`` analyses in a row agreed on the same
screen + button + place, at most one every ``tap_cooldown_s``. PLAY / Try Again need lives >= 1
read on the HUD; lives 0 = stop. Stops (with the frame saved by bot.py): a screen that stays
unknown / an ad without ✕ / a known screen with nothing allowed for too long, the same button
tapped ``same_button_max`` times without effect, ``max_taps_between_levels`` screen taps without
getting back into a level.

Ad chains (user, run 20261020-224631: skip -> App Store sheet -> end card): ✕ / skip taps do not
count towards ``same_button_max`` / ``max_taps_between_levels``; they repeat every
``ad_retap_s`` for as many ads as come, until a known screen (gameplay, fail, menu, ...) is
back. Only ``ad_chain_max_s`` without one stops the bot. Never Download / INSTALL / Get: taps
are only ✕ / skip glyphs in the corner zones.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import FlowConfig, SessionConfig
from screens import FORBIDDEN, ScreenResult

LIVES_BUTTONS = ("play", "try_again")
AD_CLOSE = "ad_close"
AD_SKIP = "ad_skip"
AD_BUTTONS = (AD_CLOSE, AD_SKIP)


@dataclass
class ScreenAction:
    kind: str                                  # tap | wait | stop
    screen: str
    reason: str
    button: str = ""
    rel: Optional[Tuple[float, float]] = None
    score: float = 0.0


def _inside(p: Tuple[float, float], box: Tuple[float, float, float, float], m: float) -> bool:
    return box[0] - m <= p[0] <= box[2] + m and box[1] - m <= p[1] <= box[3] + m


def plan(res: ScreenResult, img: np.ndarray, lives: Optional[int], fc: FlowConfig,
         ad_recent: bool = False) -> ScreenAction:
    """The one action allowed on this non-gameplay frame (pure; no state, no tap)."""
    s = res.screen

    def button(*names: str) -> Optional[ScreenAction]:
        for n in names:
            m = res.find(n)
            if m is not None:
                return ScreenAction("tap", s, f"{n} {m.score:.2f}", n, m.center_rel, m.score)
        return None

    act: Optional[ScreenAction] = None
    if s == "level_complete":
        act = button("next")
    elif s == "menu":
        act = button("close_red", "play")
    elif s == "revive":
        act = button("quit", "close_red")
    elif s == "fail":
        act = button("try_again")
    elif s == "ad" and res.find("take_a_break") is not None:
        return ScreenAction("wait", s, "take a break: the ad follows")
    if s == "ad" or (s == "other" and ad_recent):
        from adclose import find_close, find_skip

        for mark in find_close(img, fc):
            act = ScreenAction("tap", s, f"ad ✕ {mark.score:.2f}", AD_CLOSE, mark.center_rel,
                               mark.score)
            break
        if act is None:
            for mark in find_skip(img, fc):
                act = ScreenAction("tap", s, f"ad {'skip' if mark.kind == 'skip' else '✕ on a disc'}"
                                   f" {mark.score:.2f}", AD_SKIP, mark.center_rel, mark.score)
                break
    if act is None:
        return ScreenAction("wait", s, "unknown screen" if s == "other" else
                            "ad: no ✕ yet" if s == "ad" else "no allowed button")
    for name in FORBIDDEN:
        m = res.find(name)
        if m is not None and _inside(act.rel, m.box_rel, fc.forbidden_margin):
            return ScreenAction("wait", s, f"{act.button} lies on forbidden {name} - not tapped")
    if act.button in LIVES_BUTTONS:
        if lives == 0:
            return ScreenAction("stop", s, "lives 0")
        if lives is None and fc.require_lives:
            return ScreenAction("wait", s, f"{act.button}: lives not read")
    return act


class ScreenFlow:
    """Confirmation, cooldown and stop rules around ``plan``. ``feed`` -> the action to carry
    out now: ``tap`` (confirmed, cooldown over), ``stop`` or ``wait``."""

    def __init__(self, fc: FlowConfig, dry_run: bool = False) -> None:
        self.fc, self.dry_run = fc, dry_run
        self.streak: Optional[ScreenAction] = None
        self.count = 0
        self.last_tap = -1e9
        self.since: Optional[Tuple[str, float]] = None    # screen type, waiting since
        self.same: Dict[Tuple[str, str], int] = {}        # (screen, button) -> taps, screen kept
        self.taps_between = 0
        self.last_ad = -1e9
        self.chain_start: Optional[float] = None          # first ad tap of the current chain
        self.dry_done: set = set()                        # (screen, button) logged in a dry run

    def in_gameplay(self) -> None:
        self.streak, self.count, self.since = None, 0, None
        self.same.clear()
        self.dry_done.clear()
        self.taps_between = 0
        self.chain_start = None

    def _same_place(self, a: ScreenAction, b: ScreenAction) -> bool:
        if (a.kind, a.screen, a.button) != (b.kind, b.screen, b.button):
            return False
        if a.rel is None or b.rel is None:
            return a.rel is b.rel
        tol = self.fc.target_tolerance
        return abs(a.rel[0] - b.rel[0]) <= tol and abs(a.rel[1] - b.rel[1]) <= tol

    def _timeout(self, screen: str) -> float:
        fc = self.fc
        return {"other": fc.unknown_max_s, "ad": fc.ad_max_s}.get(screen, fc.stuck_max_s)

    def feed(self, res: ScreenResult, img: np.ndarray, lives: Optional[int],
             now: float) -> ScreenAction:
        if res.screen == "ad":
            self.last_ad = now
        if lives == 0:
            return ScreenAction("stop", res.screen, "lives 0")
        if res.screen not in ("ad", "other"):
            self.chain_start = None                       # a known screen: the ad chain ended
        if self.chain_start is not None and now - self.chain_start > self.fc.ad_chain_max_s:
            return ScreenAction("stop", res.screen, f"ad chain for {now - self.chain_start:.0f}s "
                                "without a known screen")
        ad_recent = (now - self.last_ad <= self.fc.ad_context_s
                     or now - self.last_tap <= self.fc.ad_after_tap_s)
        act = plan(res, img, lives, self.fc, ad_recent)
        if act.kind == "stop":
            return act
        if self.since is None or self.since[0] != res.screen:
            self.since = (res.screen, now)
            self.same = {k: v for k, v in self.same.items() if k[0] == res.screen}
        if self.streak is not None and self._same_place(self.streak, act):
            self.count += 1
        else:
            self.streak, self.count = act, 1
        waited = now - self.since[1]
        if act.kind == "tap" and self.count >= self.fc.confirm_frames:
            ad = act.button in AD_BUTTONS
            cooldown = max(self.fc.tap_cooldown_s, self.fc.ad_retap_s if ad else 0.0)
            if now - self.last_tap < cooldown:
                return ScreenAction("wait", act.screen, "cooldown after the last screen tap")
            key = (act.screen, act.button)
            if self.dry_run and key in self.dry_done:
                act = ScreenAction("wait", act.screen, f"{act.button} (dry run: logged once)")
            elif ad:
                return act                                # ad chains: as many as come
            elif self.same.get(key, 0) >= self.fc.same_button_max:
                return ScreenAction("stop", act.screen, f"{act.button} tapped "
                                    f"{self.same[key]}x and the screen did not change")
            elif self.taps_between >= self.fc.max_taps_between_levels:
                return ScreenAction("stop", act.screen, f"{self.taps_between} screen taps "
                                    "without getting back into a level")
            else:
                return act
        if waited > self._timeout(res.screen):
            return ScreenAction("stop", res.screen, f"screen {res.screen} for {waited:.0f}s "
                                f"({act.reason})")
        return act if act.kind == "wait" else ScreenAction("wait", act.screen,
                                                           f"confirming {act.reason}")

    def tapped(self, act: ScreenAction, now: float) -> None:
        key = (act.screen, act.button)
        self.last_tap = now
        self.streak, self.count = None, 0
        self.since = (act.screen, now)
        if self.dry_run:
            self.dry_done.add(key)
            return
        if act.button in AD_BUTTONS:
            if self.chain_start is None:
                self.chain_start = now
            return
        self.same[key] = self.same.get(key, 0) + 1
        self.taps_between += 1


@dataclass
class Session:
    """Levels played in one ``python -m bot --levels N`` run and the unattended limits."""
    levels: int
    sc: SessionConfig
    results: List[str] = None   # type: ignore[assignment]
    fails_in_a_row: int = 0

    def __post_init__(self) -> None:
        self.levels = min(self.levels, self.sc.max_levels)
        self.results = []

    def record(self, result: str) -> Optional[str]:
        """A level ended (win / fail); returns why the session stops now, or None."""
        self.results.append(result)
        self.fails_in_a_row = self.fails_in_a_row + 1 if result == "fail" else 0
        if len(self.results) >= self.levels:
            return f"{len(self.results)} level(s) played (--levels {self.levels})"
        if self.fails_in_a_row >= self.sc.max_consecutive_fails:
            return f"{self.fails_in_a_row} levels lost in a row"
        return None

    def summary(self) -> Dict[str, Any]:
        return {"levels": len(self.results), "wins": self.results.count("win"),
                "fails": self.results.count("fail"), "results": list(self.results)}


def draw_screen_tap(img: np.ndarray, act: ScreenAction, res: ScreenResult) -> np.ndarray:
    """The frame with the tapped point (green) and forbidden anchors (red boxes)."""
    import cv2

    out = img.copy()
    h, w = out.shape[:2]
    for name in FORBIDDEN:
        m = res.find(name)
        if m is not None:
            x0, y0, x1, y1 = m.box_rel
            cv2.rectangle(out, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)),
                          (0, 0, 255), 2)
    if act.rel is not None:
        p = (int(act.rel[0] * w), int(act.rel[1] * h))
        cv2.circle(out, p, 18, (0, 255, 0), 3)
        cv2.putText(out, f"{act.screen}: {act.button}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
    return out
