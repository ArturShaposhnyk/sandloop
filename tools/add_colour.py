"""Add a new colour to the palette from saved crops, then rebuild the palette.

    python -m tools.add_colour lime dataset/unknown_colours/*-bucket-r0c3.png
    python -m tools.add_colour lime dataset/unknown_colours/*-sand-89_-61_61.png
    python -m tools.add_colour gold crop.png --group keys

The live overlay saves every colour that matches nothing in the palette to
dataset/unknown_colours/ (``<time>-bucket|lock|key-r<r>c<c>.png`` cell crops and
``<time>-sand-<L>_<a>_<b>.png`` pixel patches). The group (buckets / sand / keys) comes from the
file name unless ``--group`` is given. Colours go to dataset/extra_colours.json, which
``python -m tools.build_palette`` merges into dataset/palette.json, so they survive rebuilds.
Prefer annotating a dataset screenshot when you can (dataset/annotations.json): that also gives
tests. Check the result with ``python -m vision <frame>``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from palette import Lab, Palette, median_lab

GROUPS = ("buckets", "sand", "keys")
KIND_GROUP = {"bucket": "buckets", "sand": "sand", "lock": "keys", "key": "keys"}


def group_of(path: Path) -> Optional[str]:
    """Group from a dataset/unknown_colours file name (``<date>-<time>-<ms>-<kind>-...``)."""
    parts = path.stem.split("-")
    return KIND_GROUP.get(parts[3]) if len(parts) > 3 else None


def colours_from_crop(img: np.ndarray, group: str, palette: Palette, v) -> List[Lab]:
    """LAB looks of the colour in one crop: a bucket cell gives its top and body part, a sand
    patch / icon its median."""
    if group == "buckets":
        from cells import cell_features

        f = cell_features(img, palette, v)
        looks = [lab for lab in (f.top_lab, f.body_lab) if lab is not None]
        if looks:
            return looks
    if group == "keys":
        from cells import icon_lab

        lab = icon_lab(img)
        if lab is not None:
            return [lab]
    return [median_lab(img.reshape(-1, 3))]


def add_extra(path: Path, group: str, name: str, labs: Sequence[Lab]) -> Dict:
    """Append ``labs`` as looks of colour ``name`` in ``group`` to the extra-colours file."""
    data: Dict = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    looks = data.setdefault(group, {}).setdefault(name, [])
    looks.extend([round(float(x), 2) for x in lab] for lab in labs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return data


def merge_extra(pal: Palette, path: Path) -> List[str]:
    """Add the colours of the extra-colours file to ``pal`` (used by build_palette)."""
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    added = []
    for group in GROUPS:
        for name, looks in data.get(group, {}).items():
            getattr(pal, group).setdefault(name, []).extend(
                tuple(float(x) for x in lab) for lab in looks)
            added.append(f"{group}/{name}")
    return added


def main(argv: Optional[Sequence[str]] = None) -> int:
    import cv2

    from config import load_config

    ap = argparse.ArgumentParser(prog="python -m tools.add_colour", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="colour name, e.g. green, darkgreen")
    ap.add_argument("files", nargs="+", type=Path, help="crops from dataset/unknown_colours/")
    ap.add_argument("--group", choices=GROUPS, default=None,
                    help="palette group (default: from the file name)")
    ap.add_argument("--no-rebuild", action="store_true", help="only update extra_colours.json")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    if not args.name.isidentifier():
        print(f"error: colour name must be a plain word, got {args.name!r}", file=sys.stderr)
        return 2
    cfg = load_config(args.config)
    palette = Palette.load(cfg.paths.palette)
    by_group: Dict[str, List[Lab]] = {}
    for f in args.files:
        group = args.group or group_of(f)
        if group is None:
            print(f"error: {f}: cannot tell the group from the name, pass --group",
                  file=sys.stderr)
            return 2
        img = cv2.imread(str(f))
        if img is None or img.size == 0:
            print(f"error: cannot read {f}", file=sys.stderr)
            return 1
        by_group.setdefault(group, []).extend(colours_from_crop(img, group, palette, cfg.vision))
    extra = Path(cfg.paths.dataset) / "extra_colours.json"
    for group, labs in by_group.items():
        add_extra(extra, group, args.name, labs)
        for lab in labs:
            near, d = palette.nearest(lab, group)
            note = f" (nearest existing: {near}, dist {d:.1f})" if near else ""
            print(f"{group}/{args.name}: LAB {'/'.join(f'{x:.0f}' for x in lab)}{note}")
    print(f"saved to {extra}")
    if args.no_rebuild:
        return 0
    from tools import build_palette

    return build_palette.main(["--config", args.config] if args.config else [])


if __name__ == "__main__":
    sys.exit(main())
