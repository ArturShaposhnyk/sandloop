"""Screen classification: gameplay / level_complete / fail / revive / ad / menu / other.

Anchors (buttons, headers, ✕, the k/5 counter) are matched as colour templates cut from the
dataset (dataset/templates/, built by tools/build_palette.py) on a frame resized to
``vision.canonical_width``. Rules are data: first rule whose anchors are found wins.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import VisionConfig

SCREEN_TYPES = ("gameplay", "level_complete", "fail", "revive", "ad", "menu", "other")

# Anchors the bot must never tap: coins, purchases, ads, store links (GAME_RULES "Bot behaviour
# outside gameplay"). No tap may land inside one of their boxes (flow.py).
FORBIDDEN = ("keep_playing", "watch_ad", "get_app", "appstore_badge")

# (screen type, anchors — any one of them is enough). Order matters: overlays first.
# status_icons = the iOS status bar (signal + wifi): the game hides it, ads show it (020).
RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("revive", ("out_of_slot", "keep_playing", "quit")),
    ("fail", ("level_failed", "try_again")),
    ("level_complete", ("level_completed", "next")),
    ("ad", ("take_a_break", "get_app", "appstore_badge", "ad_mute", "status_icons")),
    ("other", ("logo",)),
    ("gameplay", ("counter_slash5",)),
    ("menu", ("play", "close_red")),
]


@dataclass
class Match:
    name: str
    score: float
    center_rel: Tuple[float, float]   # where it was found (relative screen coordinates)
    box_rel: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)   # x0, y0, x1, y1


@dataclass
class ScreenResult:
    screen: str
    matches: Dict[str, Match] = field(default_factory=dict)   # anchors found above threshold

    def find(self, name: str) -> Optional[Match]:
        return self.matches.get(name)


def load_templates(directory: Path) -> Dict[str, np.ndarray]:
    import cv2

    out = {}
    for p in sorted(Path(directory).glob("*.png")):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None:
            out[p.stem] = img
    if not out:
        raise FileNotFoundError(f"no screen templates in {directory} — build them with "
                                "`python -m tools.build_palette`")
    return out


def popup_bar(img: np.ndarray, v: VisionConfig) -> bool:
    """A popup sliding in over gameplay (run 20261021-101655 frame 00507: the revive popup's
    green "Keep Playing" matched only 0.49 while it moved): a bright green bar at least
    ``popup_bar_min_run`` of the screen width wide in at least ``popup_bar_min_rows`` of its
    rows (lower two thirds). No bucket / button of the board is that wide."""
    import cv2

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img[h // 3:], cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] >= v.popup_bar_hue[0]) & (hsv[..., 0] <= v.popup_bar_hue[1])
         & (hsv[..., 1] >= v.popup_bar_min_sv[0]) & (hsv[..., 2] >= v.popup_bar_min_sv[1]))
    rows = 0
    for row in m[m.sum(axis=1) >= v.popup_bar_min_run * w]:
        d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        if (np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max() >= v.popup_bar_min_run * w:
            rows += 1
    return rows >= v.popup_bar_min_rows * h


class ScreenClassifier:
    def __init__(self, templates: Dict[str, np.ndarray], v: VisionConfig) -> None:
        self.templates = templates
        self.v = v

    def _gray(self, img: np.ndarray) -> np.ndarray:
        import cv2

        k = self.v.canonical_width / img.shape[1]
        return cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)

    def match(self, gray: np.ndarray, name: str) -> Match:
        import cv2

        tpl = self.templates[name]
        best = Match(name, -1.0, (0.0, 0.0))
        gh, gw = gray.shape[:2]
        for s in self.v.template_scales:
            t = tpl if s == 1.0 else cv2.resize(tpl, None, fx=s, fy=s,
                                                interpolation=cv2.INTER_AREA)
            if t.shape[0] > gray.shape[0] or t.shape[1] > gray.shape[1] or min(t.shape[:2]) < 4:
                continue
            res = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)
            _, score, _, (x, y) = cv2.minMaxLoc(res)
            if score > best.score:
                th, tw = t.shape[:2]
                best = Match(name, float(score), ((x + tw / 2) / gw, (y + th / 2) / gh),
                             (x / gw, y / gh, (x + tw) / gw, (y + th) / gh))
        return best

    def classify(self, img: np.ndarray) -> ScreenResult:
        gray = self._gray(img)
        if float(gray.mean()) < 12:
            return ScreenResult("other")
        found: Dict[str, Match] = {}
        for name in self.templates:
            m = self.match(gray, name)
            if m.score >= self.v.template_thresholds.get(name, self.v.template_threshold):
                found[name] = m
        for screen, anchors in RULES:
            if any(a in found for a in anchors):
                return ScreenResult(screen, found)
        return ScreenResult("other", found)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import cv2

    from config import load_config

    ap = argparse.ArgumentParser(prog="python -m screens", description="Classify screenshots.")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    clf = ScreenClassifier(load_templates(Path(cfg.paths.dataset) / "templates"), cfg.vision)
    for path in args.images:
        img = cv2.imread(path)
        if img is None:
            print(f"{path}: cannot read")
            continue
        res = clf.classify(img)
        anchors = ", ".join(f"{m.name} {m.score:.2f}" for m in res.matches.values())
        print(f"{Path(path).name}: {res.screen}  [{anchors}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
