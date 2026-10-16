"""Greedy move choice (docs/GAME_RULES.md, "Strategy implications").

Hard rules, a candidate is rejected if any fails:
  * it is tappable: path rule AND visual face-up agree (never UNKNOWN, locked or disagreeing);
  * the conveyor has a free slot ("k/5" counter read and k < 5);
  * its colour's sand is exposed at the bottom of the picture right now (a stricter threshold
    when the tap would take the last free slot: keep a slot usable, avoid deadlock);
  * enough of its colour remains to fill it: sand >= margin x (conveyor need + 1) x capacity.
Score of the rest: exposed share of the colour + buckets / lock cells the tap connects to the
exit (one-tap lookahead). Ties go to the front row, then the left.

    python -m strategy dataset/screens/*.png   # decisions on saved shots -> debug/strategy-*.png
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

from config import StrategyConfig
from state import LevelState, Pos

if TYPE_CHECKING:
    import numpy as np

    from vision import GameView


@dataclass
class Candidate:
    pos: Pos
    color: str
    score: float = 0.0
    rejected: Optional[str] = None             # why it can't be played now (None = allowed)
    opens: List[Pos] = field(default_factory=list)       # buckets connected by this tap
    opens_locks: List[Pos] = field(default_factory=list)

    def describe(self) -> str:
        r, c = self.pos
        if self.rejected:
            return f"r{r}c{c} {self.color}: no - {self.rejected}"
        extra = f", opens {len(self.opens)} bucket(s)" if self.opens else ""
        extra += f", {len(self.opens_locks)} lock cell(s)" if self.opens_locks else ""
        return f"r{r}c{c} {self.color}: score {self.score:.2f}{extra}"


@dataclass
class Decision:
    move: Optional[Candidate]                  # None = wait
    reason: str
    candidates: List[Candidate] = field(default_factory=list)


def choose(state: LevelState, sc: StrategyConfig) -> Decision:
    free = state.free_slots
    if free is None:
        return Decision(None, "wait: conveyor counter not read")
    cands: List[Candidate] = []
    cap = state.capacity_estimate()
    for pos in state.tappable:
        cell = state.cell(pos)
        cand = Candidate(pos, cell.color or "?")
        cands.append(cand)
        exposed = state.exposed.get(cand.color, 0.0)
        min_exp = sc.last_slot_min_exposed if free == 1 else sc.min_exposed
        need = (state.conveyor_need(cand.color) + 1) * cap * sc.sufficiency_margin if cap else 0.0
        have = state.sand.get(cand.color, 0.0)
        if free == 0:
            cand.rejected = "conveyor full"
        elif exposed < min_exp:
            cand.rejected = (f"sand not exposed ({exposed:.0%} < {min_exp:.0%}"
                             f"{', last slot' if free == 1 else ''})")
        elif have < need:
            cand.rejected = f"not enough sand ({have:.1%} < {need:.1%})"
        if cand.rejected:
            continue
        cand.opens, cand.opens_locks = state.opened_by(pos)
        cand.score = (sc.weight_exposed * exposed + sc.weight_open_bucket * len(cand.opens)
                      + sc.weight_open_lock * len(cand.opens_locks))
    allowed = [c for c in cands if c.rejected is None]
    if not allowed:
        if free == 0:
            why = "conveyor full"
        elif not cands:
            why = "no tappable bucket"
        else:
            why = "no bucket passes the rules"
        return Decision(None, f"wait: {why}", cands)
    best = max(allowed, key=lambda c: (round(c.score, 6), -c.pos[0], -c.pos[1]))
    r, c = best.pos
    return Decision(best, f"tap r{r}c{c} {best.color} (score {best.score:.2f})", cands)


# ---------------------------------------------------------------------------------------------
# Drawing (overlay --dry-run, CLI)
# ---------------------------------------------------------------------------------------------


def draw_decision(img: "np.ndarray", gv: "GameView", decision: Decision) -> "np.ndarray":
    """vision.draw + the decision: the chosen bucket gets a thick cyan ring and "TAP", rejected
    candidates an orange cross, the decision text sits below the grid."""
    import cv2

    from vision import draw

    out = draw(img, gv)
    h, w = out.shape[:2]
    lat = gv.lattice
    for cand in decision.candidates:
        x0, y0, x1, y1 = (int(round(a)) for a in lat.rect(*cand.pos))
        if cand.rejected:
            cv2.line(out, (x0 + 6, y0 + 6), (x1 - 6, y1 - 6), (0, 140, 255), 2)
            cv2.line(out, (x1 - 6, y0 + 6), (x0 + 6, y1 - 6), (0, 140, 255), 2)
    if decision.move is not None:
        x0, y0, x1, y1 = lat.rect(*decision.move.pos)
        cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
        rad = int(0.45 * min(x1 - x0, y1 - y0))
        cv2.circle(out, (cx, cy), rad, (0, 0, 0), 7)
        cv2.circle(out, (cx, cy), rad, (255, 255, 0), 4)
        _label(out, "TAP", (cx - 18, cy + 6), 0.7)
        for p in decision.move.opens:
            bx0, by0, bx1, by1 = (int(round(a)) for a in lat.rect(*p))
            cv2.rectangle(out, (bx0 + 4, by0 + 4), (bx1 - 4, by1 - 4), (255, 255, 0), 1)
    lines = [decision.reason] + [c.describe() for c in decision.candidates][:5]
    y0 = int(lat.y0 + lat.rows * lat.ch) + 6
    y1 = min(h, y0 + 20 * len(lines) + 6)
    out[y0:y1] = (out[y0:y1] * 0.25).astype(out.dtype)
    for i, line in enumerate(lines):
        y = y0 + 18 + 20 * i
        if y > h - 4:
            break
        cv2.putText(out, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 0) if i == 0 else (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _label(img: "np.ndarray", s: str, org: Tuple[int, int], scale: float) -> None:
    import cv2

    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------------------------
# Frame -> decision (shared by the overlay and the CLI; the bot will use it too)
# ---------------------------------------------------------------------------------------------


@dataclass
class Analysis:
    screen: str
    view: Optional["GameView"] = None
    state: Optional[LevelState] = None
    decision: Optional[Decision] = None
    error: Optional[str] = None

    def tap_rel(self) -> Optional[Tuple[float, float]]:
        """Relative screen point of the chosen move (what a tap would hit), or None."""
        if self.view is None or self.decision is None or self.decision.move is None:
            return None
        return self.view.cell_center_rel(*self.decision.move.pos)


class Analyzer:
    """Loads palette, digits and screen templates once; ``analyze(frame)`` reads a frame of the
    iPhone screen and decides the next move. Never taps."""

    def __init__(self, cfg) -> None:
        from digits import DigitReader
        from palette import Palette
        from screens import ScreenClassifier, load_templates

        self.cfg = cfg
        self.palette = Palette.load(cfg.paths.palette)
        self.reader = DigitReader.load(Path(cfg.paths.dataset) / "digits.npz")
        self.classifier = ScreenClassifier(load_templates(Path(cfg.paths.dataset) / "templates"),
                                           cfg.vision)

    def analyze(self, img: "np.ndarray", log=print) -> Analysis:
        from vision import VisionError, read_gameplay

        screen = self.classifier.classify(img).screen
        if screen != "gameplay":
            return Analysis(screen)
        try:
            gv = read_gameplay(img, self.palette, self.cfg.vision, self.reader, log=log)
        except VisionError as e:
            return Analysis(screen, error=str(e))
        st = self.cfg.strategy
        state = LevelState.from_view(gv, st.conveyor_slots, st.unknown_fill_need)
        return Analysis(screen, gv, state, choose(state, st))


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    import cv2

    from config import load_config

    ap = argparse.ArgumentParser(prog="python -m strategy",
                                 description="Greedy decisions on saved screenshots (never taps).")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    an = Analyzer(cfg)
    out_dir = Path(cfg.paths.debug)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.images:
        img = cv2.imread(path)
        if img is None:
            print(f"{path}: cannot read")
            continue
        res = an.analyze(img, log=lambda s: print("  " + s))
        print(f"== {Path(path).name}: {res.screen}")
        if res.error:
            print(f"  vision error: {res.error}")
        if res.state is None or res.decision is None:
            continue
        for line in res.state.summary():
            print("  " + line)
        for cand in res.decision.candidates:
            print("    " + cand.describe())
        tap = res.tap_rel()
        print(f"  => {res.decision.reason}"
              + (f"  at rel ({tap[0]:.3f}, {tap[1]:.3f})" if tap else ""))
        p = out_dir / f"strategy-{Path(path).stem}.png"
        cv2.imwrite(str(p), draw_decision(img, res.view, res.decision))
        print(f"  saved {p}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
