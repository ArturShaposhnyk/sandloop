"""Sand in the picture: share per colour, the contact layer the belt touches and the lower
part of the picture (split out of vision.py)."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from config import VisionConfig
from palette import Palette, bgr_to_lab, lab_distance
from zones import crop_rel

Above = Dict[str, Dict[str, float]]   # contact colour -> colour above it -> contact share


def read_sand(img: np.ndarray, palette: Palette, v: VisionConfig
              ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Above]:
    """Share of the picture covered by each sand colour, the share of each colour in the
    contact layer (sand the belt can collect right now, see ``contact_layers``), in the lower
    ``lower_band`` of the picture zone (sand that will reach the belt soon) and what lies
    right above each contact colour (``contact_layers``). The picture is read down to its real
    bottom edge (``_bottom_edge``): it can reach below the configured zone, and the colour that
    settles at the bottom was cut off (run 20261020-171133, L187 end: darkred 0.17 % in the zone
    vs 1.4 % to the edge - "not enough sand" for the darkred bucket at 26 % of the contact
    layer)."""
    edge = _bottom_edge(img, palette, v)
    if edge is not None and edge > int(v.picture[3] * img.shape[0]):
        h, w = img.shape[:2]
        pic = img[int(v.picture[1] * h):edge, int(v.picture[0] * w):int(v.picture[2] * w)]
    else:
        pic = crop_rel(img, v.picture)
    lab = bgr_to_lab(pic)
    idx, names = palette.classify_pixels(lab, "sand", v.sand_max_dist)
    if "picture_empty" in palette.ui:
        idx[lab_distance(lab, palette.ui["picture_empty"]) < 10] = -1
    total = idx.size
    sand = {n: float((idx == i).sum()) / total for i, n in enumerate(names)}
    layers = contact_layers(img, palette, v)
    exposed, above = layers if layers is not None else (None, {})
    if exposed is None:   # bottom edge not found: fall back to the zone's bottom band
        band = idx[int((1 - v.bottom_band) * idx.shape[0]):]
        exposed = {n: float((band == i).sum()) / max(1, band.size) for i, n in enumerate(names)}
    low = idx[int((1 - v.lower_band) * idx.shape[0]):]
    lower = {n: float((low == i).sum()) / max(1, low.size) for i, n in enumerate(names)}
    return ({n: f for n, f in sand.items() if f > v.sand_min_share},
            {n: f for n, f in exposed.items() if f > 0.01},
            {n: f for n, f in lower.items() if f > 0.01}, above)


def contact_layer(img: np.ndarray, palette: Palette, v: VisionConfig
                  ) -> Optional[Dict[str, float]]:
    """The contact layer alone (``contact_layers``)."""
    layers = contact_layers(img, palette, v)
    return None if layers is None else layers[0]


def contact_layers(img: np.ndarray, palette: Palette, v: VisionConfig
                   ) -> Optional[Tuple[Dict[str, float], Above]]:
    """(contact, above). ``contact``: colour shares of the sand a belt bucket can collect: per
    picture column the lowest ``contact_layer`` px of sand above the picture's real bottom edge
    (a bucket takes the bottom grains plus a few above, GAME_RULES.md; empty picture pixels are
    skipped, so a column whose bottom was already collected reaches further up, within
    ``contact_reach``). The edge is found by walking from the zone's bottom while rows are
    still sand / empty picture (within ``contact_search``). None if no edge is found.
    ``above[c][d]``: share of the contact layer that becomes colour ``d`` once the contact sand
    of colour ``c`` is collected - per column, ``c``'s contact pixels times the colour mix of
    the next ``contact_layer`` px of sand above (sand falls down: what the belt uncovers;
    strategy lookahead). Columns with no sand above add nothing."""
    h, w = img.shape[:2]
    x0, x1 = int(v.picture[0] * w), int(v.picture[2] * w)
    edge = _bottom_edge(img, palette, v)
    if edge is None:
        return None
    layer, reach = max(1, int(v.contact_layer * h)), max(1, int(v.contact_reach * h))
    lab = bgr_to_lab(img[max(0, edge - reach - layer):edge, x0:x1])
    idx, names = palette.classify_pixels(lab, "sand", v.sand_max_dist)
    if "picture_empty" in palette.ui:
        idx[lab_distance(lab, palette.ui["picture_empty"]) < 10] = -1
    sand = idx >= 0
    from_bottom = np.cumsum(sand[::-1], axis=0)[::-1]   # sand px at or below, per column
    in_reach = np.zeros_like(sand)
    in_reach[max(0, sand.shape[0] - reach):] = True
    take = sand & in_reach & (from_bottom <= layer)
    total = int(take.sum())
    if total == 0:
        return {}, {}
    nxt = sand & ~take & (from_bottom > layer) & (from_bottom <= 2 * layer)
    contact = {n: float(((idx == i) & take).sum()) / total for i, n in enumerate(names)}
    above: Above = {}
    n_next = nxt.sum(axis=0).astype(float)
    has_next = n_next > 0
    for i, c in enumerate(names):
        cc = ((idx == i) & take).sum(axis=0).astype(float)   # c's contact px per column
        if not cc[has_next].any():
            continue
        row = {}
        for j, d in enumerate(names):
            nd = ((idx == j) & nxt).sum(axis=0).astype(float)
            share = float((cc[has_next] * nd[has_next] / n_next[has_next]).sum()) / total
            if share > 0.002:
                row[d] = share
        if row:
            above[c] = row
    return contact, above


def _bottom_edge(img: np.ndarray, palette: Palette, v: VisionConfig) -> Optional[int]:
    """Frame row just below the picture's lowest row (None if not found within
    ``contact_search`` of the configured zone bottom)."""
    h, w = img.shape[:2]
    x0, x1 = int(v.picture[0] * w), int(v.picture[2] * w)
    yb = int(v.picture[3] * h)
    reach = max(1, int(v.contact_search * h))
    ya, yz = max(0, yb - reach), min(h, yb + reach)
    lab = bgr_to_lab(img[ya:yz, x0:x1])
    idx, _ = palette.classify_pixels(lab, "sand", v.sand_max_dist)
    pictured = idx >= 0
    if "picture_empty" in palette.ui:
        pictured |= lab_distance(lab, palette.ui["picture_empty"]) < 10
    inside = pictured.mean(axis=1) >= 0.5
    y = yb - ya
    if not inside[min(y, len(inside) - 1)]:
        while y > 0 and not inside[y - 1]:   # zone reaches below the picture: walk up
            y -= 1
        if y == 0:
            return None
    else:
        while y < len(inside) and inside[y]:   # picture goes on below the zone: walk down
            y += 1
        if y == len(inside):
            return None
    return ya + y
