"""Lock / key icons in grid cells, recognised by shape (see docs/GAME_RULES.md, L183).

Templates are grey cell crops cut from annotated dataset cells by tools/build_palette.py
(dataset/templates/icons/<kind>-<colour>-*.png); the icon's colour comes from the palette's
``keys`` group (cells.icon_lab).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

ICON_SIZE = 48   # icon templates: grey cell crops resized to this square
ICON_PAD = 4     # the template is the crop minus this border, so the icon may shift a little


def icon_grey(cell_bgr: np.ndarray) -> np.ndarray:
    import cv2

    grey = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(grey, (ICON_SIZE, ICON_SIZE), interpolation=cv2.INTER_AREA)


@dataclass
class IconMatcher:
    """Lock / key icons by shape: grey templates cut from annotated cells by
    tools/build_palette.py (dataset/templates/icons/<kind>-<colour>-*.png). Colour-independent
    (a yellow lock template matches a red lock at ~0.85); the colour comes from the palette."""
    templates: List[Tuple[str, np.ndarray]] = field(default_factory=list)   # (kind, grey)
    shells: Optional["ShellMatcher"] = None   # L188 shell icons on buckets (../shells/)
    picture: Any = None                       # picture.PictureMatcher (../picture/, L188)

    def match(self, cell_bgr: np.ndarray) -> Tuple[Optional[str], float]:
        import cv2

        if not self.templates or cell_bgr.size == 0:
            return None, 0.0
        grey = icon_grey(cell_bgr).astype(np.float32)
        best, best_s = None, -1.0
        for kind, t in self.templates:
            inner = t[ICON_PAD:-ICON_PAD, ICON_PAD:-ICON_PAD].astype(np.float32)
            sc = float(cv2.matchTemplate(grey, inner, cv2.TM_CCOEFF_NORMED).max())
            if sc > best_s:
                best, best_s = kind, sc
        return best, best_s

    @classmethod
    def load(cls, directory: Path, palette: Any = None) -> "IconMatcher":
        """Also loads ../shells/ (shell icons on buckets) and ../picture/ (shells / keyholes in
        the picture; its shell masks need ``palette``)."""
        import cv2

        out = cls()
        for p in sorted(Path(directory).glob("*.png")):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None and img.shape == (ICON_SIZE, ICON_SIZE):
                out.templates.append((p.stem.split("-")[0], img))
        shells = Path(directory).parent / "shells"
        if shells.is_dir():
            out.shells = ShellMatcher.load(shells)
        pic = Path(directory).parent / "picture"
        if pic.is_dir():
            from picture import PictureMatcher

            out.picture = PictureMatcher.load(pic, palette)
        return out


SHELL_SIZE = 96                       # shell templates: grey bucket crops resized to this square
SHELL_BOX = (0.02, 0.42, 0.25, 0.75)  # the shell inside the crop: y0, y1, x0, x1 (template part)
SHELL_SEARCH = 0.55                   # ... searched in the crop's upper part (rows)


def _gradient(grey: np.ndarray) -> np.ndarray:
    import cv2

    g = grey.astype(np.float32)
    m = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    return m / (float(m.max()) + 1e-6)


def shell_grey(cell_bgr: np.ndarray) -> np.ndarray:
    import cv2

    grey = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(grey, (SHELL_SIZE, SHELL_SIZE), interpolation=cv2.INTER_AREA)


@dataclass
class ShellMatcher:
    """L188 shell icon on a bucket (grey shell inside an open bucket, a shell of the bucket's
    colour on a dome), by shape: the shell's gradient image (colour-independent) matched in the
    upper part of the bucket crop. Templates: dataset/templates/shells/ (tools/build_palette.py,
    annotated '...+shell' cells). Annotated shells score >= 0.82 with their own template left
    out, plain buckets <= 0.78; tower tops (a disc on a dome) up to 0.83 - cells.classify_cell
    tells them apart by colour."""
    templates: List[Tuple[str, np.ndarray]] = field(default_factory=list)   # (name, gradient)

    def match(self, cell_bgr: np.ndarray) -> float:
        import cv2

        if not self.templates or cell_bgr.size == 0:
            return 0.0
        g = _gradient(shell_grey(cell_bgr))[:int(SHELL_SEARCH * SHELL_SIZE)]
        return max(float(cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED).max())
                   for _, t in self.templates)

    @classmethod
    def load(cls, directory: Path) -> "ShellMatcher":
        import cv2

        out = cls()
        y0, y1, x0, x1 = (int(f * SHELL_SIZE) for f in SHELL_BOX)
        for p in sorted(Path(directory).glob("*.png")):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None and img.shape == (SHELL_SIZE, SHELL_SIZE):
                out.templates.append((p.stem, _gradient(img)[y0:y1, x0:x1]))
        return out
