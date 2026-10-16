"""Find the iPhone Mirroring window and grab frames of the iPhone screen area.

This module (together with actions.py) is the only place where relative coordinates (0..1 of the
iPhone screen) are converted to capture pixels or global screen points. macOS-only modules
(Quartz, mss) are imported lazily so the pure logic is testable on any OS.

CLI:
    python -m capture list      # all on-screen windows (owner, title, id, bounds)
    python -m capture info      # resolved window, detected iPhone screen rect, scale factor
    python -m capture shot      # save the window and iPhone-screen frames to debug/
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import Config, load_config

# ---------------------------------------------------------------------------------------------
# Geometry (pure)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px <= self.right and self.y <= py <= self.bottom


@dataclass(frozen=True)
class ScreenGeometry:
    """Where the iPhone screen is.

    window:  window bounds, global screen points (top-left origin, as CGWindowList/CGEvent use).
    content: iPhone screen area, global screen points.
    scale:   capture pixels per point (2.0 on Retina; detected at runtime).
    """

    window: Rect
    content: Rect
    scale: float

    def _check_rel(self, rx: float, ry: float) -> None:
        if not (0.0 <= rx <= 1.0 and 0.0 <= ry <= 1.0):
            raise ValueError(f"relative coordinate out of range 0..1: ({rx}, {ry})")

    def rel_to_point(self, rx: float, ry: float) -> Tuple[float, float]:
        self._check_rel(rx, ry)
        return self.content.x + rx * self.content.w, self.content.y + ry * self.content.h

    def point_to_rel(self, x: float, y: float) -> Tuple[float, float]:
        return (x - self.content.x) / self.content.w, (y - self.content.y) / self.content.h

    @property
    def frame_size(self) -> Tuple[int, int]:
        """(width, height) in pixels of a frame of the content area."""
        return int(round(self.content.w * self.scale)), int(round(self.content.h * self.scale))

    def rel_to_pixel(self, rx: float, ry: float) -> Tuple[int, int]:
        """Relative -> pixel in a content frame, clamped to valid indices."""
        self._check_rel(rx, ry)
        w, h = self.frame_size
        return min(int(rx * w), w - 1), min(int(ry * h), h - 1)

    def pixel_to_rel(self, px: float, py: float) -> Tuple[float, float]:
        w, h = self.frame_size
        return px / w, py / h


def content_from_window_pixels(window: Rect, px_rect: Rect, scale: float) -> Rect:
    """Convert a rect in window-capture pixels into global points."""
    return Rect(window.x + px_rect.x / scale, window.y + px_rect.y / scale,
                px_rect.w / scale, px_rect.h / scale)


def content_from_relative(window: Rect, rel: Sequence[float]) -> Rect:
    x, y, w, h = rel
    return Rect(window.x + x * window.w, window.y + y * window.h, window.w * w, window.h * h)


def content_from_alpha(alpha: np.ndarray, threshold: int = 250) -> Optional[Rect]:
    """Bounding box (pixels) of the opaque area of a window-only capture, or None if none."""
    mask = alpha >= threshold
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    return Rect(float(cols[0]), float(rows[0]), float(cols[-1] - cols[0] + 1),
                float(rows[-1] - rows[0] + 1))


def detect_content_rect(img: np.ndarray, max_trim: float = 0.15, tolerance: float = 6.0) -> Rect:
    """Find the iPhone screen area inside a window capture (pixels).

    Trims border strips (title bar, frame, margins) that are near-uniform in colour and match the
    colour of the outermost strip on that side. Never trims more than ``max_trim`` per side.
    """
    gray = img if img.ndim == 2 else np.asarray(img[..., :3], dtype=np.float32).mean(axis=2)
    gray = gray.astype(np.float32)
    h, w = gray.shape

    def trim(lines: np.ndarray, limit: int) -> int:
        # lines: (n, k) — strips ordered from the outer edge inwards
        edge = float(np.median(lines[0]))
        n = 0
        while n < limit:
            line = lines[n]
            if float(line.std()) > tolerance or abs(float(line.mean()) - edge) > tolerance * 2:
                break
            n += 1
        return n

    top = trim(gray, int(h * max_trim))
    bottom = trim(gray[::-1], int(h * max_trim))
    left = trim(gray.T, int(w * max_trim))
    right = trim(gray.T[::-1], int(w * max_trim))
    return Rect(float(left), float(top), float(w - left - right), float(h - top - bottom))


# ---------------------------------------------------------------------------------------------
# Frame stability (pure)
# ---------------------------------------------------------------------------------------------


def frame_change(a: np.ndarray, b: np.ndarray, size: int = 96) -> float:
    """Mean absolute difference between two frames, 0..1, on a small grey thumbnail."""
    import cv2

    def prep(f: np.ndarray) -> np.ndarray:
        if f.ndim == 3:
            f = cv2.cvtColor(f[..., :3], cv2.COLOR_BGR2GRAY)
        return cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)

    return float(np.abs(prep(a) - prep(b)).mean() / 255.0)


def wait_for_stable(grab: Callable[[], np.ndarray], threshold: float, timeout_s: float,
                    interval_s: float, clock: Callable[[], float] = time.monotonic,
                    sleep: Callable[[float], None] = time.sleep) -> Tuple[np.ndarray, bool]:
    """Grab frames until two consecutive ones differ by < threshold. Returns (frame, stable)."""
    deadline = clock() + timeout_s
    prev = grab()
    while True:
        sleep(interval_s)
        cur = grab()
        if frame_change(prev, cur) < threshold:
            return cur, True
        if clock() >= deadline:
            return cur, False
        prev = cur


# ---------------------------------------------------------------------------------------------
# Window lookup (macOS)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowInfo:
    window_id: int
    owner: str
    title: str
    pid: int
    layer: int
    bounds: Rect


class WindowNotFound(RuntimeError):
    pass


def list_windows(on_screen_only: bool = True) -> List[WindowInfo]:
    import Quartz

    opts = Quartz.kCGWindowListOptionOnScreenOnly if on_screen_only else Quartz.kCGWindowListOptionAll
    opts |= Quartz.kCGWindowListExcludeDesktopElements
    raw = Quartz.CGWindowListCopyWindowInfo(opts, Quartz.kCGNullWindowID) or []
    return [_window_from_dict(d) for d in raw]


def _window_from_dict(d: Dict[str, Any]) -> WindowInfo:
    b = d.get("kCGWindowBounds", {}) or {}
    return WindowInfo(
        window_id=int(d.get("kCGWindowNumber", 0)),
        owner=str(d.get("kCGWindowOwnerName", "") or ""),
        title=str(d.get("kCGWindowName", "") or ""),
        pid=int(d.get("kCGWindowOwnerPID", 0)),
        layer=int(d.get("kCGWindowLayer", 0)),
        bounds=Rect(float(b.get("X", 0)), float(b.get("Y", 0)),
                    float(b.get("Width", 0)), float(b.get("Height", 0))),
    )


def pick_window(windows: Sequence[WindowInfo], owner_name: str,
                title_contains: Optional[str] = None) -> WindowInfo:
    """Largest normal-layer window whose owner matches (case-insensitive)."""
    owner_l = owner_name.casefold()
    cands = [w for w in windows if w.owner.casefold() == owner_l and w.layer == 0
             and w.bounds.w > 50 and w.bounds.h > 50]
    if title_contains:
        cands = [w for w in cands if title_contains.casefold() in w.title.casefold()]
    if not cands:
        raise WindowNotFound(
            f"no on-screen window owned by {owner_name!r}. Is iPhone Mirroring open and visible? "
            "Run `python -m capture list` and set window.owner_name in config.yaml.")
    return max(cands, key=lambda w: w.bounds.w * w.bounds.h)


# ---------------------------------------------------------------------------------------------
# Grabbing (mss)
# ---------------------------------------------------------------------------------------------


class CaptureError(RuntimeError):
    pass


def crop_points(img: np.ndarray, origin: Rect, r: Rect, scale: float) -> np.ndarray:
    """Crop rect ``r`` (global points) out of an image of ``origin`` (global points)."""
    x0, y0 = int(round((r.x - origin.x) * scale)), int(round((r.y - origin.y) * scale))
    x1, y1 = x0 + int(round(r.w * scale)), y0 + int(round(r.h * scale))
    h, w = img.shape[:2]
    return img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]


class Grabber:
    """Returns BGR uint8 frames in physical pixels.

    backend "window": Quartz capture of the mirroring window alone (full Retina resolution, not
    obscured by windows on top, alpha channel marks the transparent margins).
    backend "mss": screen-region capture via mss (always nominal resolution on macOS, sees
    whatever is on top). Fallback only.
    """

    def __init__(self, backend: str = "window") -> None:
        if backend not in ("window", "mss"):
            raise ValueError(f"unknown capture backend {backend!r}")
        self.backend = backend
        self._sct = None
        if backend == "mss":
            import mss

            self._sct = mss.MSS() if hasattr(mss, "MSS") else mss.mss()

    def _grab_window_quartz(self, win: WindowInfo) -> Tuple[np.ndarray, np.ndarray]:
        import Quartz

        img = Quartz.CGWindowListCreateImage(
            Quartz.CGRectNull, Quartz.kCGWindowListOptionIncludingWindow, win.window_id,
            Quartz.kCGWindowImageBoundsIgnoreFraming | Quartz.kCGWindowImageBestResolution)
        if img is None:
            raise CaptureError(f"could not capture window {win.window_id} (minimized or closed?)")
        w, h = Quartz.CGImageGetWidth(img), Quartz.CGImageGetHeight(img)
        bpr = Quartz.CGImageGetBytesPerRow(img)
        data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(img))
        # 32-bit little-endian, alpha first => BGRA byte order in memory
        bgra = np.frombuffer(data, np.uint8).reshape(h, bpr)[:, : w * 4].reshape(h, w, 4)
        return np.ascontiguousarray(bgra[..., :3]), np.ascontiguousarray(bgra[..., 3])

    def _grab_points_mss(self, r: Rect) -> np.ndarray:
        region = {"left": int(round(r.x)), "top": int(round(r.y)),
                  "width": max(1, int(round(r.w))), "height": max(1, int(round(r.h)))}
        return np.ascontiguousarray(np.asarray(self._sct.grab(region))[..., :3])

    def grab_window(self, win: WindowInfo) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Whole window: (BGR image, alpha or None)."""
        if self.backend == "window":
            return self._grab_window_quartz(win)
        return self._grab_points_mss(win.bounds), None

    def grab_content(self, win: WindowInfo, geo: ScreenGeometry) -> np.ndarray:
        """Only the iPhone screen area."""
        if self.backend == "window":
            img, _ = self._grab_window_quartz(win)
            return np.ascontiguousarray(crop_points(img, geo.window, geo.content, geo.scale))
        return self._grab_points_mss(geo.content)

    def close(self) -> None:
        if self._sct is not None:
            self._sct.close()


def resolve_geometry(cfg: Config, grabber: Grabber,
                     windows: Optional[Sequence[WindowInfo]] = None) -> Tuple[WindowInfo, ScreenGeometry]:
    win = pick_window(windows if windows is not None else list_windows(),
                      cfg.window.owner_name, cfg.window.title_contains)
    img, alpha = grabber.grab_window(win)
    scale = img.shape[1] / win.bounds.w
    if isinstance(cfg.screen.content, str):
        px = content_from_alpha(alpha) if alpha is not None else None
        if px is None:
            px = detect_content_rect(img, cfg.screen.auto_max_trim, cfg.screen.uniform_tolerance)
        content = content_from_window_pixels(win.bounds, px, scale)
    else:
        content = content_from_relative(win.bounds, cfg.screen.content)
    return win, ScreenGeometry(win.bounds, content, scale)


class WindowTracker:
    """Caches the window geometry and re-resolves it every ``refresh_interval_s`` seconds."""

    def __init__(self, cfg: Config, grabber: Grabber,
                 resolver: Optional[Callable[[], Tuple[WindowInfo, ScreenGeometry]]] = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._interval = cfg.window.refresh_interval_s
        self._resolver = resolver or (lambda: resolve_geometry(cfg, grabber))
        self._clock = clock
        self._grabber = grabber
        self._last = -float("inf")
        self.window: Optional[WindowInfo] = None
        self.geometry: Optional[ScreenGeometry] = None

    def get(self, force: bool = False) -> ScreenGeometry:
        now = self._clock()
        if force or self.geometry is None or now - self._last >= self._interval:
            self.window, self.geometry = self._resolver()
            self._last = now
        return self.geometry

    def grab(self) -> np.ndarray:
        """Grab the iPhone screen area (BGR, physical pixels)."""
        geo = self.get()
        return self._grabber.grab_content(self.window, geo)


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def _fmt_rect(r: Rect) -> str:
    return f"x={r.x:.0f} y={r.y:.0f} w={r.w:.0f} h={r.h:.0f}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m capture", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["list", "info", "shot"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--all", action="store_true", help="list: include off-screen windows")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "list":
        wins = list_windows(on_screen_only=not args.all)
        print(f"{'id':>6}  {'pid':>6}  {'layer':>5}  {'bounds (points)':<32}  owner  |  title")
        for w in wins:
            print(f"{w.window_id:>6}  {w.pid:>6}  {w.layer:>5}  {_fmt_rect(w.bounds):<32}  "
                  f"{w.owner}  |  {w.title}")
        if wins and not any(w.title for w in wins):
            print("\nNote: no window titles visible — Screen Recording permission is probably "
                  "missing (run `python -m permissions`).")
        return 0

    grabber = Grabber(cfg.screen.capture)
    try:
        win, geo = resolve_geometry(cfg, grabber)
    except (WindowNotFound, CaptureError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"window : id={win.window_id} owner={win.owner!r} title={win.title!r} pid={win.pid}")
    print(f"bounds : {_fmt_rect(geo.window)} (points)")
    mode = "auto" if isinstance(cfg.screen.content, str) else "manual"
    print(f"screen : {_fmt_rect(geo.content)} (points)  mode={mode} capture={grabber.backend}")
    print(f"scale  : {geo.scale:.3f} px/pt   frame={geo.frame_size[0]}x{geo.frame_size[1]} px")
    if args.command == "shot":
        import cv2
        from pathlib import Path

        out = Path(cfg.paths.debug)
        out.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        win_img = grabber.grab_window(win)[0].copy()
        s = geo.scale
        x0, y0 = (geo.content.x - geo.window.x) * s, (geo.content.y - geo.window.y) * s
        cv2.rectangle(win_img, (int(x0), int(y0)),
                      (int(x0 + geo.content.w * s), int(y0 + geo.content.h * s)), (0, 255, 0), 2)
        cv2.imwrite(str(out / f"window-{stamp}.png"), win_img)
        frame = grabber.grab_content(win, geo)
        cv2.imwrite(str(out / f"screen-{stamp}.png"), frame)
        if float(frame.max()) < 8:
            print("warning: capture is all black — Screen Recording permission is likely missing.")
        print(f"saved  : {out}/window-{stamp}.png (green = detected screen), {out}/screen-{stamp}.png")
    grabber.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
