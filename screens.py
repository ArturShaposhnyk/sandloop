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

# (screen type, anchors — any one of them is enough). Order matters: overlays first.
RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("revive", ("out_of_slot", "keep_playing", "quit")),
    ("fail", ("level_failed", "try_again")),
    ("ad", ("take_a_break", "get_app", "appstore_badge", "ad_mute")),
    ("other", ("logo",)),
    ("gameplay", ("counter_slash5",)),
    ("menu", ("play", "close_red")),
]


@dataclass
class Match:
    name: str
    score: float
    center_rel: Tuple[float, float]   # where it was found (relative screen coordinates)


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
        for s in self.v.template_scales:
            t = tpl if s == 1.0 else cv2.resize(tpl, None, fx=s, fy=s,
                                                interpolation=cv2.INTER_AREA)
            if t.shape[0] > gray.shape[0] or t.shape[1] > gray.shape[1] or min(t.shape[:2]) < 4:
                continue
            res = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)
            _, score, _, (x, y) = cv2.minMaxLoc(res)
            if score > best.score:
                cx = (x + t.shape[1] / 2) / gray.shape[1]
                cy = (y + t.shape[0] / 2) / gray.shape[0]
                best = Match(name, float(score), (cx, cy))
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
