"""Find the iPhone Mirroring window and grab frames of it.

CLI:
  python -m capture list            # all on-screen windows: owner, title, id, bounds
  python -m capture info            # resolved window, scale factor, detected iPhone screen rect
  python -m capture shot out.png    # save one cropped iPhone-screen frame

macOS-only modules (Quartz, mss) are imported lazily so pure helpers stay testable anywhere.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from config import Config, ContentDetectConfig, load_config
from geometry import FULL_WINDOW, Rect, ScreenGeometry, detect_scale


class CaptureError(RuntimeError):
    pass


@dataclass(frozen=True)
class WindowInfo:
    window_id: int
    owner: str
    title: str
    pid: int
    bounds: Rect  # global points
    layer: int


# --------------------------------------------------------------------------- window list
def list_windows(on_screen_only: bool = True) -> List[WindowInfo]:
    import Quartz  # lazy: macOS only

    opts = Quartz.kCGWindowListExcludeDesktopElements
    if on_screen_only:
        opts |= Quartz.kCGWindowListOptionOnScreenOnly
    raw = Quartz.CGWindowListCopyWindowInfo(opts, Quartz.kCGNullWindowID) or []
    out: List[WindowInfo] = []
    for w in raw:
        b = w.get("kCGWindowBounds") or {}
        out.append(
            WindowInfo(
                window_id=int(w.get("kCGWindowNumber", 0)),
                owner=str(w.get("kCGWindowOwnerName") or ""),
                title=str(w.get("kCGWindowName") or ""),
                pid=int(w.get("kCGWindowOwnerPID", 0)),
                bounds=Rect(float(b.get("X", 0)), float(b.get("Y", 0)),
                            float(b.get("Width", 0)), float(b.get("Height", 0))),
                layer=int(w.get("kCGWindowLayer", 0)),
            )
        )
    return out


def pick_window(windows: List[WindowInfo], owner_name: str,
                title_contains: str = "") -> Optional[WindowInfo]:
    """Largest normal-layer window whose owner matches (case-insensitive)."""
    owner_l, title_l = owner_name.casefold(), title_contains.casefold()
    cands = [
        w for w in windows
        if w.layer == 0 and w.owner.casefold() == owner_l
        and (not title_l or title_l in w.title.casefold())
        and w.bounds.w > 50 and w.bounds.h > 50
    ]
    if not cands:
        return None
    return max(cands, key=lambda w: w.bounds.w * w.bounds.h)


def find_window(cfg: Config) -> WindowInfo:
    win = pick_window(list_windows(), cfg.window.owner_name, cfg.window.title_contains)
    if win is None:
        raise CaptureError(
            f"No on-screen window owned by '{cfg.window.owner_name}'. Is iPhone Mirroring open "
            "and not minimized? Run `python -m capture list` and set window.owner_name in "
            "config.yaml to the exact owner name shown there (it may be localized)."
        )
    return win


# --------------------------------------------------------------------------- grabbing
def _grab_mss(bounds: Rect) -> np.ndarray:
    import mss  # lazy

    left, top, width, height = bounds.as_int()
    factory = getattr(mss, "MSS", None) or mss.mss
    with factory() as sct:
        shot = sct.grab({"left": left, "top": top, "width": width, "height": height})
        return np.asarray(shot)[..., :3].copy()  # BGRA -> BGR


def _grab_quartz(window_id: int) -> np.ndarray:
    import Quartz  # lazy

    img = Quartz.CGWindowListCreateImage(
        Quartz.CGRectNull,
        Quartz.kCGWindowListOptionIncludingWindow,
        window_id,
        Quartz.kCGWindowImageBoundsIgnoreFraming | Quartz.kCGWindowImageBestResolution,
    )
    if img is None:
        raise CaptureError("CGWindowListCreateImage returned nothing (permission or window gone)")
    w, h = Quartz.CGImageGetWidth(img), Quartz.CGImageGetHeight(img)
    bpr = Quartz.CGImageGetBytesPerRow(img)
    data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(img))
    arr = np.frombuffer(data, dtype=np.uint8).reshape(h, bpr // 4, 4)[:, :w, :3]
    return arr.copy()  # BGRA (little-endian premultiplied first) -> BGR


def grab_window(win: WindowInfo, backend: str) -> np.ndarray:
    frame = _grab_quartz(win.window_id) if backend == "quartz" else _grab_mss(win.bounds)
    if frame.size == 0:
        raise CaptureError("captured an empty frame")
    return frame


# --------------------------------------------------------------------------- content rect
def detect_content_rect(img: np.ndarray, opts: ContentDetectConfig) -> Tuple[Rect, str]:
    """Heuristically find the iPhone screen inside a window image.

    Assumes window chrome (title bar / frame) is a roughly uniform color band along an edge.
    Each edge is handled independently: if its outermost line is uniform, lines are trimmed
    inward while most of their pixels still match that color. Returns (rect as fractions of
    the window, human-readable reason). Falls back to the full window.
    """
    h, w = img.shape[:2]
    if h < 20 or w < 20:
        return FULL_WINDOW, "image too small; using full window"
    px = img[..., :3].astype(np.int16)
    tol = opts.border_tolerance

    def band(lines: np.ndarray) -> int:
        """lines: (n, len, 3) from the edge inward. Return how many leading lines are chrome."""
        edge = lines[0]
        chrome = np.median(edge, axis=0)
        if float(np.median(np.abs(edge - chrome).max(axis=1))) > tol:
            return 0  # edge line not uniform: no chrome on this side
        match = (np.abs(lines - chrome).max(axis=2) <= tol).mean(axis=1) > 0.5
        n = int(np.argmin(match)) if not match.all() else len(match)
        return n

    top = band(px)
    bottom = band(px[::-1])
    left = band(px.transpose(1, 0, 2))
    right = band(px.transpose(1, 0, 2)[::-1])
    y0, y1, x0, x1 = top, h - bottom, left, w - right
    if y1 - y0 < h * 0.3 or x1 - x0 < w * 0.3:
        return FULL_WINDOW, "window looks blank or mostly chrome; using full window"
    if (y0, y1, x0, x1) == (0, h, 0, w):
        return FULL_WINDOW, "no uniform window chrome found; using full window"
    aspect = (x1 - x0) / max(1, (y1 - y0))
    if not (opts.min_aspect <= aspect <= opts.max_aspect):
        return FULL_WINDOW, (f"detected area aspect {aspect:.3f} outside "
                             f"[{opts.min_aspect}, {opts.max_aspect}]; using full window")
    rect = Rect(x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h)
    return rect, f"detected screen area {x1 - x0}x{y1 - y0}px (aspect {aspect:.3f})"


# --------------------------------------------------------------------------- tracker
class WindowTracker:
    """Keeps the window geometry fresh (re-resolved every refresh_interval_s) and grabs frames."""

    def __init__(self, cfg: Config, clock: Callable[[], float] = time.monotonic) -> None:
        self.cfg = cfg
        self._clock = clock
        self._win: Optional[WindowInfo] = None
        self._geom: Optional[ScreenGeometry] = None
        self._last_resolve = -1e9
        self._content_size: Optional[Tuple[float, float]] = None
        self.content_reason = ""

    @property
    def window(self) -> WindowInfo:
        self._maybe_refresh()
        assert self._win is not None
        return self._win

    def _maybe_refresh(self, force: bool = False) -> None:
        now = self._clock()
        if force or self._win is None or now - self._last_resolve >= self.cfg.window.refresh_interval_s:
            self._win = find_window(self.cfg)
            self._last_resolve = now
            size = (self._win.bounds.w, self._win.bounds.h)
            if self._geom is not None and size != self._content_size:
                self._geom = None  # resized: recompute scale + content rect on next frame
            elif self._geom is not None:
                self._geom = ScreenGeometry(self._win.bounds, self._geom.scale,
                                            self._geom.content_frac)

    def _build_geometry(self, frame: np.ndarray) -> ScreenGeometry:
        assert self._win is not None
        scale = detect_scale(frame.shape[1], self._win.bounds.w)
        override = self.cfg.window.content_rect
        if override is not None:
            frac, self.content_reason = Rect(*override), "manual override from config"
        else:
            frac, self.content_reason = detect_content_rect(frame, self.cfg.window.content_detect)
        self._content_size = (self._win.bounds.w, self._win.bounds.h)
        return ScreenGeometry(self._win.bounds, scale, frac)

    def grab(self) -> Tuple[np.ndarray, ScreenGeometry]:
        """Return (cropped iPhone-screen BGR frame, geometry used for it)."""
        self._maybe_refresh()
        assert self._win is not None
        frame = grab_window(self._win, self.cfg.capture.backend)
        if self._geom is None:
            self._geom = self._build_geometry(frame)
        x, y, w, h = self._geom.content_px.as_int()
        crop = frame[max(0, y):y + h, max(0, x):x + w]
        return crop, self._geom

    def geometry(self) -> ScreenGeometry:
        if self._geom is None:
            self.grab()
        assert self._geom is not None
        self._maybe_refresh()
        return self._geom


# --------------------------------------------------------------------------- CLI
def _cmd_list(_: Config) -> int:
    wins = list_windows()
    print(f"{'OWNER':28} {'TITLE':30} {'ID':>7} {'PID':>7}  LAYER  BOUNDS (x, y, w, h in points)")
    for w in sorted(wins, key=lambda w: (w.layer, w.owner.casefold())):
        b = w.bounds
        print(f"{w.owner[:28]:28} {w.title[:30]:30} {w.window_id:7d} {w.pid:7d}  {w.layer:5d}  "
              f"{b.x:.0f}, {b.y:.0f}, {b.w:.0f}, {b.h:.0f}")
    if wins and not any(w.title for w in wins):
        print("\nNote: no window has a title — Screen Recording is probably not granted to this "
              "terminal. Run `python -m permissions`.")
    return 0


def _cmd_info(cfg: Config) -> int:
    tr = WindowTracker(cfg)
    crop, g = tr.grab()
    w = tr.window
    print(f"window     : '{w.owner}' id={w.window_id} pid={w.pid}")
    print(f"bounds (pt): {g.window_pt}")
    print(f"scale      : {g.scale} px/pt")
    print(f"content    : frac={g.content_frac}  ({tr.content_reason})")
    print(f"content pt : {g.content_pt}")
    print(f"crop size  : {crop.shape[1]}x{crop.shape[0]} px")
    return 0


def _cmd_shot(cfg: Config, out: str) -> int:
    import cv2

    from pathlib import Path

    crop, _ = WindowTracker(cfg).grab()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out, crop)
    print(f"saved {crop.shape[1]}x{crop.shape[0]} frame to {out}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m capture", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--backend", choices=("mss", "quartz"), help="override capture.backend")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="print all on-screen windows")
    sub.add_parser("info", help="show resolved window + geometry")
    p_shot = sub.add_parser("shot", help="save one cropped frame")
    p_shot.add_argument("out", nargs="?", default="debug/shot.png")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.backend:
        cfg.capture.backend = args.backend
    try:
        if args.cmd == "list":
            return _cmd_list(cfg)
        if args.cmd == "info":
            return _cmd_info(cfg)
        return _cmd_shot(cfg, args.out)
    except CaptureError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
