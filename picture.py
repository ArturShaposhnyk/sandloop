"""Objects in the picture (L188, user 2026-10-21; docs/GAME_RULES.md "Picture locks", "Shells").

* **Shells** (``PicShell``): a shell of a bucket colour sits in the sand and falls with it; a
  shell bucket on the belt collects nothing until a shell of its colour reaches the picture's
  bottom and drops into it. Found by the shape of their outline (gradient templates from
  dataset/templates/picture/shell-*.png, colour-independent); colour = the saturated pixels of
  the box centre against the palette's bucket colours; ``depth`` = distance of the shell's
  bottom to the picture's bottom edge (share of the frame height; 0 = at the bottom).
* **Picture locks** (``PicLock``): a box with a keyhole (L188 bottom left / right). The sand
  behind it does not fall and cannot be collected until the key of its colour flies (keys sit
  on the board, L188 under the curtain); sand above it rests on it. Found by the keyhole
  (templates keyhole-*.png), the box by its dark grey frame around the keyhole, colour = the
  keyhole ring against the palette's ``keys``. The box's pixels are seen through a light
  overlay (colours not readable): ``mask`` keeps them out of the sand reading.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from config import VisionConfig
from palette import Palette, bgr_to_lab

Rect = Tuple[float, float, float, float]
SHELL_COLOUR_DIST = 30.0              # LAB: a shell pixel's distance to its bucket colour


@dataclass
class PicShell:
    colour: Optional[str]
    x: float                          # centre, relative to the screen
    y: float
    depth: float                      # shell bottom -> picture bottom edge, share of height
    score: float = 0.0


@dataclass
class PicLock:
    colour: Optional[str]
    rect: Rect                        # the box, relative [x0, y0, x1, y1]
    score: float = 0.0


def _gradient(grey: np.ndarray) -> np.ndarray:
    import cv2

    g = cv2.medianBlur(grey, 5).astype(np.float32)
    m = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3),
                      cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    return m / (float(m.max()) + 1e-6)


def _peaks(score: np.ndarray, threshold: float, radius: int) -> List[Tuple[int, int, float]]:
    """(x, y, score) of local maxima above ``threshold``, at least ``radius`` apart."""
    s = score.copy()
    out = []
    while True:
        y, x = np.unravel_index(int(np.argmax(s)), s.shape)
        best = float(s[y, x])
        if best < threshold:
            return out
        out.append((int(x), int(y), best))
        s[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1] = -1.0


@dataclass
class Template:
    grad: np.ndarray                  # gradient image
    mask: Optional[np.ndarray] = None   # shell + rim (shells), None = the whole template
    body: Optional[np.ndarray] = None   # shell pixels (bool)


@dataclass
class PictureMatcher:
    shells: List[Template] = field(default_factory=list)
    keyholes: List[Template] = field(default_factory=list)

    @classmethod
    def load(cls, directory: Path, palette: Optional[Palette] = None) -> "PictureMatcher":
        """``directory``: grey crops shell-<colour>-*.png / keyhole-<colour>-*.png
        (tools/build_palette.py); a shell's mask needs the colour crop next to it
        (``*.bgr.png``) and the palette."""
        import cv2

        out = cls()
        for p in sorted(Path(directory).glob("*.png")):
            if p.stem.endswith(".bgr"):
                continue
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if p.stem.startswith("keyhole"):
                out.keyholes.append(Template(_gradient(img)))
            elif p.stem.startswith("shell"):
                colour = p.stem.split("-")[1]
                bgr = cv2.imread(str(p.with_suffix(".bgr.png")))
                if bgr is None or palette is None or colour not in palette.buckets:
                    continue
                body = _blob(bgr, palette, colour)
                rim = cv2.dilate(body.astype(np.uint8), np.ones((5, 5), np.uint8))
                out.shells.append(Template(_gradient(img), rim.astype(np.float32), body))
        return out

    def _match(self, grad: np.ndarray, temps: List[Template], threshold: float):
        import cv2

        hits: List[Tuple[int, int, float, Tuple[int, int], Template]] = []
        for tt in temps:
            t = tt.grad
            if t.shape[0] > grad.shape[0] or t.shape[1] > grad.shape[1]:
                continue
            if tt.mask is not None:
                res = cv2.matchTemplate(grad, t, cv2.TM_CCORR_NORMED, mask=tt.mask)
                res[~np.isfinite(res)] = 0.0
            else:
                res = cv2.matchTemplate(grad, t, cv2.TM_CCOEFF_NORMED)
            for x, y, s in _peaks(res, threshold, max(t.shape) // 4):
                hits.append((x, y, s, t.shape, tt))
        hits.sort(key=lambda h: -h[2])
        kept: List[Tuple[int, int, float, Tuple[int, int], Template]] = []
        for h in hits:   # one hit per object over all templates
            if all(abs(h[0] - k[0]) > h[3][1] // 3 or abs(h[1] - k[1]) > h[3][0] // 3
                   for k in kept):
                kept.append(h)
        return kept

    def find(self, img: np.ndarray, palette: Palette, v: VisionConfig, bottom: Optional[int],
             colours: Optional[set] = None) -> Tuple[List[PicShell], List[PicLock]]:
        """Shells and picture locks in the picture (``bottom``: the picture's bottom edge row,
        sand._bottom_edge). ``colours``: colours of the shell buckets in play - only shells of
        these colours count (None: any bucket colour)."""
        import cv2

        h, w = img.shape[:2]
        # widened by half a template: shells at the picture's sides (L188 x 95-137) stick out
        # of the configured zone
        pad = max([t.grad.shape[1] for t in self.shells + self.keyholes], default=0) // 2
        x0, y0 = max(0, int(v.picture[0] * w) - pad), int(v.picture[1] * h)
        x1 = min(w, int(v.picture[2] * w) + pad)
        y1 = bottom if bottom is not None else int(v.picture[3] * h)
        region = img[y0:y1, x0:x1]
        if region.size == 0 or (not self.shells and not self.keyholes):
            return [], []
        grad = _gradient(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY))
        locks = []
        for x, y, s, (th, tw), _ in self._match(grad, self.keyholes, v.keyhole_threshold):
            cx, cy = x0 + x + tw // 2, y0 + y + th // 2
            rect = _box_around(img, cx, cy, tw // 2 + 2, th // 2 + 2, v)
            colour = _ring_colour(img[y0 + y:y0 + y + th, x0 + x:x0 + x + tw], palette, v)
            if rect is not None:
                locks.append(PicLock(colour, rect, round(s, 3)))
        shells = []
        if colours is not None and not colours:
            return shells, locks
        lab = bgr_to_lab(region)
        # shaded shell pixels sit up to ~30 LAB from the bucket looks (as the template masks)
        idx, names = palette.classify_pixels(lab, "buckets", SHELL_COLOUR_DIST)
        for x, y, s, (th, tw), tt in self._match(grad, self.shells, v.pic_shell_threshold):
            bx0, by0 = x0 + x, y0 + y
            if any(l.rect[0] * w <= bx0 + tw / 2 <= l.rect[2] * w
                   and l.rect[1] * h <= by0 + th / 2 <= l.rect[3] * h for l in locks):
                continue   # inside a lock box (its keyhole / overlay), never a shell
            ids = idx[y:y + th, x:x + tw][tt.body]
            ids = ids[ids >= 0]
            if ids.size == 0:
                continue
            vals, counts = np.unique(ids, return_counts=True)
            colour = names[int(vals[np.argmax(counts)])]
            purity = counts.max() / max(1, int(tt.body.sum()))
            if purity < v.pic_shell_purity or (colours is not None and colour not in colours):
                continue
            depth = max(0, y1 - (by0 + th)) / h
            shells.append(PicShell(colour, (bx0 + tw / 2) / w, (by0 + th / 2) / h,
                                   round(depth, 4), round(s, 3)))
        return shells, locks


def _blob(bgr: np.ndarray, palette: Palette, colour: str) -> np.ndarray:
    """The shell in a template crop: the largest blob of its colour (LAB within 30 of one of
    the palette's looks of it)."""
    import cv2

    lab = bgr_to_lab(bgr)
    d = np.min(np.stack([np.linalg.norm(lab - np.asarray(c), axis=-1)
                         for c in palette.buckets[colour]]), axis=0)
    n, lb, st, _ = cv2.connectedComponentsWithStats((d < SHELL_COLOUR_DIST).astype(np.uint8))
    if n < 2:
        return np.ones(bgr.shape[:2], bool)
    return lb == 1 + int(np.argmax(st[1:, 4]))


def _ring_colour(crop: np.ndarray, palette: Palette, v: VisionConfig) -> Optional[str]:
    from cells import icon_lab

    lab = icon_lab(crop)
    return None if lab is None else palette.nearest(lab, "keys", v.color_max_dist)[0]


def _box_around(img: np.ndarray, cx: int, cy: int, ox: int, oy: int,
                v: VisionConfig) -> Optional[Rect]:
    """The lock box: from just outside the keyhole (``ox`` / ``oy`` from its centre: the black
    keyhole itself is dark) outwards to the first run of the box's dark grey frame (live 065:
    ~3 px of RGB ~65-75, unsaturated) in each direction, within ``pic_lock_max`` of the
    screen width."""
    import cv2

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    frame = (hsv[..., 2] < v.pic_lock_frame_v) & (hsv[..., 1] < 60)
    far = int(v.pic_lock_max * w)

    def scan(dx: int, dy: int) -> Optional[int]:
        x, y, run = cx + dx * ox, cy + dy * oy, 0
        for _ in range(far):
            x, y = x + dx, y + dy
            if not (0 <= x < w and 0 <= y < h):
                return None
            run = run + 1 if frame[y, x] else 0
            if run >= 2:
                return x if dx else y
        return None

    left, right, top, bot = scan(-1, 0), scan(1, 0), scan(0, -1), scan(0, 1)
    if None in (left, right, top, bot):
        return None
    return (left / w, top / h, (right + 1) / w, (bot + 1) / h)


def lock_mask(shape: Tuple[int, int], locks: List[PicLock]) -> Optional[np.ndarray]:
    """Frame-size mask of the lock boxes (their sand is behind a light overlay), or None."""
    if not locks:
        return None
    h, w = shape
    m = np.zeros((h, w), bool)
    for l in locks:
        x0, y0, x1, y1 = l.rect
        m[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = True
    return m


def locked_share(shape: Tuple[int, int], locks: List[PicLock], v: VisionConfig) -> float:
    """Share of the picture zone covered by lock boxes (their sand: colour unknown)."""
    m = lock_mask(shape, locks)
    if m is None:
        return 0.0
    h, w = shape
    z = m[int(v.picture[1] * h):int(v.picture[3] * h), int(v.picture[0] * w):int(v.picture[2] * w)]
    return round(float(z.mean()), 4) if z.size else 0.0
