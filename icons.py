"""Lock / key icons in grid cells, recognised by shape (see docs/GAME_RULES.md, L183).

Templates are grey cell crops cut from annotated dataset cells by tools/build_palette.py
(dataset/templates/icons/<kind>-<colour>-*.png); the icon's colour comes from the palette's
``keys`` group (cells.icon_lab).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

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
    def load(cls, directory: Path) -> "IconMatcher":
        import cv2

        out = cls()
        for p in sorted(Path(directory).glob("*.png")):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None and img.shape == (ICON_SIZE, ICON_SIZE):
                out.templates.append((p.stem.split("-")[0], img))
        return out
