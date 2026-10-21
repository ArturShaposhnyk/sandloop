"""Collect dataset screenshots of the iPhone screen while you play.

    python -m tools.collect                  # auto: save each new screen once it settles
    python -m tools.collect --manual         # only save when you press the snap hotkey
    python -m tools.collect --label win      # tag this session's shots (e.g. start, mid, win, fail, ad)
    python -m tools.collect --max 40         # stop after 40 shots

Snap hotkey (default ctrl+alt+s, config collect.hotkey_snap) saves the current frame in both modes
and works while the mirroring window has focus. Pause/quit hotkeys from config.yaml, or Ctrl+C.

Shots go to <paths.dataset>/screens/ as PNG (iPhone screen area only, physical pixels), with one
JSON line per shot in index.jsonl there.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from capture import CaptureError, ScreenGeometry, WindowNotFound, WindowTracker, frame_change
from config import load_config

LABEL_RE = re.compile(r"^[a-z0-9_-]{1,32}$")


class AutoTrigger:
    """Decides when auto mode saves a frame.

    A frame is saved once the screen has settled (two consecutive frames differ by less than
    ``stable_threshold``), it differs from the last saved shot by at least ``change_threshold``,
    and at least ``min_interval_s`` passed since the last save.
    """

    def __init__(self, change_threshold: float, stable_threshold: float,
                 min_interval_s: float) -> None:
        self.change_threshold = change_threshold
        self.stable_threshold = stable_threshold
        self.min_interval_s = min_interval_s
        self._prev: Optional[np.ndarray] = None
        self._saved: Optional[np.ndarray] = None
        self._saved_t = -float("inf")

    def feed(self, frame: np.ndarray, now: float) -> Optional[float]:
        """Returns the change score vs. the last saved shot if this frame should be saved."""
        prev, self._prev = self._prev, frame
        if prev is None or prev.shape != frame.shape:
            return None
        if frame_change(prev, frame) >= self.stable_threshold:
            return None  # still animating / transitioning
        if now - self._saved_t < self.min_interval_s:
            return None
        change = self.change_from_saved(frame)
        return change if change >= self.change_threshold else None

    def change_from_saved(self, frame: np.ndarray) -> float:
        if self._saved is None or self._saved.shape != frame.shape:
            return 1.0
        return frame_change(self._saved, frame)

    def mark_saved(self, frame: np.ndarray, now: float) -> None:
        self._saved, self._saved_t = frame, now


@dataclass
class Shot:
    path: Path
    record: Dict[str, Any]


def shot_name(stamp: float, source: str, label: Optional[str], seq: int) -> str:
    base = time.strftime("%Y%m%d-%H%M%S", time.localtime(stamp))
    ms = int((stamp % 1) * 1000)
    parts = [f"{base}-{ms:03d}", f"{seq:03d}", label or "", source]
    return "-".join(p for p in parts if p) + ".png"


def shot_record(name: str, stamp: float, source: str, label: Optional[str], change: float,
                geo: ScreenGeometry, frame: np.ndarray) -> Dict[str, Any]:
    rect = lambda r: [round(r.x, 1), round(r.y, 1), round(r.w, 1), round(r.h, 1)]  # noqa: E731
    return {
        "file": name,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stamp)),
        "source": source,
        "label": label,
        "change": round(change, 4),
        "frame_px": [int(frame.shape[1]), int(frame.shape[0])],
        "scale": round(geo.scale, 3),
        "window_pt": rect(geo.window),
        "screen_pt": rect(geo.content),
    }


class ShotWriter:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.count = 0
        # sequence numbers go on after the folder's highest (each session restarted at 001:
        # L188's 001-007 and L190's 001-012 clashed with the dataset's shots)
        self.first = 1 + max((int(p.stem.split("-")[3]) for p in Path(out_dir).glob("*.png")
                              if len(p.stem.split("-")) > 3 and p.stem.split("-")[3].isdigit()),
                             default=0)

    def save(self, frame: np.ndarray, geo: ScreenGeometry, source: str, label: Optional[str],
             change: float, stamp: Optional[float] = None) -> Shot:
        import cv2

        stamp = time.time() if stamp is None else stamp
        self.out_dir.mkdir(parents=True, exist_ok=True)
        name = shot_name(stamp, source, label, self.first + self.count)
        path = self.out_dir / name
        if not cv2.imwrite(str(path), frame):
            raise OSError(f"could not write {path}")
        record = shot_record(name, stamp, source, label, change, geo, frame)
        with (self.out_dir / "index.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        self.count += 1
        return Shot(path, record)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import permissions
    from capture import Grabber
    from safety import Hotkeys

    ap = argparse.ArgumentParser(prog="python -m tools.collect", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manual", action="store_true", help="save only on the snap hotkey")
    ap.add_argument("--label", default=None, help="tag for this session's shots, e.g. win, ad")
    ap.add_argument("--max", type=int, default=0, help="stop after N shots (0 = no limit)")
    ap.add_argument("--out", default=None, help="output dir (default <paths.dataset>/screens)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    if args.label is not None and not LABEL_RE.match(args.label):
        ap.error("--label must be 1-32 chars of a-z, 0-9, _ or -")
    cfg = load_config(args.config)
    permissions.require(screen=True, accessibility=False)

    c = cfg.collect
    out = Path(args.out) if args.out else Path(cfg.paths.dataset) / "screens"
    writer = ShotWriter(out)
    trigger = AutoTrigger(c.change_threshold, cfg.stability.threshold, c.min_interval_s)
    snap = threading.Event()
    hotkeys = Hotkeys(cfg.safety.hotkey_pause, cfg.safety.hotkey_quit,
                      extra={c.hotkey_snap: snap.set}).start()
    grabber = Grabber(cfg.screen.capture)
    tracker = WindowTracker(cfg, grabber)
    mode = "manual" if args.manual else "auto"
    print(f"collecting ({mode}) into {out}/ — snap={c.hotkey_snap} pause={hotkeys.pause_spec} "
          f"quit={hotkeys.quit_spec}, Ctrl+C to stop")
    waiting = was_black = False
    try:
        while not hotkeys.quit.is_set() and not (args.max and writer.count >= args.max):
            if hotkeys.paused.is_set():
                time.sleep(c.poll_s)
                continue
            try:
                geo = tracker.get(force=waiting)
                frame = tracker.grab()
            except (WindowNotFound, CaptureError) as e:
                if not waiting:
                    print(f"waiting for window: {e}")
                waiting = True
                time.sleep(1.0)
                continue
            if waiting:
                print("window found")
                waiting = False
            is_black = float(frame.max()) < 8
            if is_black and not was_black:
                print("warning: frame is all black — Screen Recording permission or a locked "
                      "mirroring session?")
            was_black = is_black
            now = time.monotonic()
            source, change = None, None
            if snap.is_set():
                snap.clear()
                source, change = "snap", trigger.change_from_saved(frame)
                trigger.feed(frame, now)  # keep its previous-frame state current
            elif not args.manual:
                change = trigger.feed(frame, now)
                source = "auto" if change is not None else None
            if source is not None:
                shot = writer.save(frame, geo, source, args.label, change)
                trigger.mark_saved(frame, now)
                print(f"[{writer.count:3d}] {source:4s} change={change:.3f}  {shot.path.name}")
            time.sleep(c.poll_s)
    except KeyboardInterrupt:
        pass
    finally:
        grabber.close()
    print(f"\nsaved {writer.count} shot(s) to {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
