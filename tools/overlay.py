"""Live debug view of the iPhone screen as the bot sees it.

  python -m tools.overlay                 # live frame + FPS + relative grid
  python -m tools.overlay --backend quartz

Keys (in the overlay window): q / Esc = quit, s = save frame to debug/, g = toggle grid.
Hover the mouse over the overlay to read relative (rx, ry) coordinates — handy for picking
points for `python -m actions tap rx ry`.

Tip: with the default `mss` backend the overlay must NOT cover the iPhone Mirroring window
(mss captures the screen region). Put it beside the window, or use `--backend quartz`.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from capture import CaptureError, WindowTracker
from config import load_config

WINDOW = "sandloop overlay"


def draw_grid(img: np.ndarray, step: float) -> None:
    if step <= 0:
        return
    h, w = img.shape[:2]
    n = int(round(1 / step))
    for i in range(1, n):
        x, y = int(w * i * step), int(h * i * step)
        cv2.line(img, (x, 0), (x, h), (80, 80, 80), 1)
        cv2.line(img, (0, y), (w, y), (80, 80, 80), 1)
        cv2.putText(img, f"{i * step:.1f}", (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        cv2.putText(img, f"{i * step:.1f}", (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)


def label(img: np.ndarray, text: str, org: Tuple[int, int]) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.overlay", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--backend", choices=("mss", "quartz"), help="override capture.backend")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.backend:
        cfg.capture.backend = args.backend

    tracker = WindowTracker(cfg)
    mouse: dict = {"pos": None}

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        mouse["pos"] = (x, y)

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    show_grid = cfg.overlay.grid_step > 0
    fps, last = 0.0, time.perf_counter()
    debug_dir = Path(cfg.paths.debug)
    printed_geom = None

    try:
        while True:
            crop, geom = tracker.grab()
            if geom != printed_geom:
                print(f"window {geom.window_pt}  scale={geom.scale}  content={geom.content_frac} "
                      f"({tracker.content_reason})")
                printed_geom = geom
            now = time.perf_counter()
            dt, last = now - last, now
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps

            h, w = crop.shape[:2]
            k = min(1.0, cfg.overlay.max_height / max(1, h))
            view = cv2.resize(crop, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA)
            if show_grid:
                draw_grid(view, cfg.overlay.grid_step)
            label(view, f"{fps:5.1f} FPS  {w}x{h}px  scale {geom.scale}", (8, 22))
            if mouse["pos"] is not None:
                mx, my = mouse["pos"]
                rx, ry = mx / view.shape[1], my / view.shape[0]
                cv2.circle(view, (mx, my), 4, (0, 0, 255), 1)
                label(view, f"rel ({rx:.3f}, {ry:.3f})", (8, view.shape[0] - 10))
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("g"):
                show_grid = not show_grid
            if key == ord("s"):
                debug_dir.mkdir(parents=True, exist_ok=True)
                out = debug_dir / f"overlay_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(str(out), crop)
                print(f"saved {out}")
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    except CaptureError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
