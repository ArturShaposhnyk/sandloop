"""Build the palette, digit templates and screen templates from annotated dataset screenshots.

    python -m tools.build_palette            # reads dataset/annotations.json + dataset/screens/

Writes dataset/palette.json (bucket, sand and UI colours in LAB), dataset/digits.npz (glyph
templates) and dataset/templates/*.png (colour screen anchors at the canonical width). Re-run it after
adding annotations (e.g. new bucket colours in later levels).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import VisionConfig, load_config
from palette import Lab, Palette, bgr_to_lab, median_lab


def load_annotations(dataset: Path) -> dict:
    return json.loads((dataset / "annotations.json").read_text(encoding="utf-8"))


def shot_path(dataset: Path, seq: str) -> Path:
    hits = sorted((dataset / "screens").glob(f"*-{seq}-*.png"))
    hits = [h for h in hits if h.stem.split("-")[3] == seq]
    if not hits:
        raise FileNotFoundError(f"no screenshot with sequence {seq} in {dataset / 'screens'}")
    return hits[0]


def read_shot(dataset: Path, seq: str) -> np.ndarray:
    import cv2

    img = cv2.imread(str(shot_path(dataset, seq)))
    if img is None:
        raise OSError(f"cannot read screenshot {seq}")
    return img


def parse_label(label: str) -> Tuple[str, Optional[str], Optional[str], Optional[bool], Optional[int]]:
    """'.', 'x', 'ice17', 'cur12', 'red-up', 'cream/red-down', 'lock:red', 'key:yellow' ->
    (kind, color, body, face_up, counter)."""
    for kind in ("lock", "key"):
        if label.startswith(kind + ":"):
            return kind, label[len(kind) + 1:], None, None, None
    if label == ".":
        return "empty", None, None, None, None
    if label == "x":
        return "none", None, None, None, None
    for prefix, kind in (("ice", "ice"), ("cur", "curtain")):
        if label.startswith(prefix):
            num = label[len(prefix):]
            return kind, None, None, None, int(num) if num else None
    color, face = label.rsplit("-", 1)
    top, _, body = color.partition("/")
    return "bucket", top, body or None, face == "up", None


def grid_labels(ann: dict) -> Dict[str, List[List[str]]]:
    return {seq: [row.split() for row in rows] for seq, rows in ann["grids"].items()}


def build_ui(dataset: Path, ann: dict, v: VisionConfig) -> Palette:
    from vision import cell_crop, find_lattice

    pal = Palette()
    for name, samples in ann["ui_colors"].items():
        pix = [crop_rel_px(read_shot(dataset, seq), rect).reshape(-1, 3) for seq, rect in samples]
        pal.ui[name] = median_lab(np.concatenate(pix))
    empty = []
    for seq, rows in grid_labels(ann).items():
        img = read_shot(dataset, seq)
        lat = find_lattice(img, pal, v)
        for r, row in enumerate(rows):
            for c, lab in enumerate(row):
                if lab == "." and lat.mask[r, c]:
                    x0, y0, x1, y1 = lat.rect(r, c)
                    cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, lat.cw / 5, lat.ch / 5
                    empty.append(img[int(cy - ry):int(cy + ry), int(cx - rx):int(cx + rx)].reshape(-1, 3))
    pal.ui["empty_cell"] = median_lab(np.concatenate(empty))
    return pal


def crop_rel_px(img: np.ndarray, r: Sequence[float]) -> np.ndarray:
    h, w = img.shape[:2]
    return img[int(r[1] * h):int(r[3] * h), int(r[0] * w):int(r[2] * w)]


def build_buckets(dataset: Path, ann: dict, pal: Palette, v: VisionConfig) -> None:
    from cells import cell_features
    from vision import cell_crop, find_lattice

    samples: Dict[Tuple[str, str], List[Lab]] = {}
    for seq, rows in grid_labels(ann).items():
        img = read_shot(dataset, seq)
        lat = find_lattice(img, pal, v)
        for r, row in enumerate(rows):
            for c, lab in enumerate(row):
                kind, color, body, face_up, _ = parse_label(lab)
                if kind != "bucket" or body is not None:
                    continue
                f = cell_features(cell_crop(img, lat, r, c, v), pal, v)
                face = "up" if face_up else "down"
                for part, val in (("top", f.top_lab), ("body", f.body_lab)):
                    if val is not None:
                        samples.setdefault((color, f"{face}-{part}"), []).append(val)
    for (color, _), labs in sorted(samples.items()):
        pal.buckets.setdefault(color, []).append(
            tuple(float(x) for x in np.median(np.asarray(labs), axis=0)))


def build_sand(dataset: Path, ann: dict, pal: Palette, v: VisionConfig,
               shots: Sequence[str] = ("001", "031", "053")) -> None:
    """Cluster the picture pixels of full-picture shots into as many colours as that level has
    (``sand_colors`` of the shot, else its bucket colours), and pair clusters with bucket
    colours (min total LAB distance)."""
    import cv2

    grids = grid_labels(ann)
    for seq in shots:
        colors = ann.get("sand_colors", {}).get(seq) or sorted(
            {parse_label(l)[1] for row in grids[seq] for l in row
             if parse_label(l)[0] == "bucket"})
        pic = crop_rel_px(read_shot(dataset, seq), v.picture)
        lab = bgr_to_lab(pic).reshape(-1, 3)
        if "picture_empty" in pal.ui:
            lab = lab[np.linalg.norm(lab - np.asarray(pal.ui["picture_empty"]), axis=1) >= 10]
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        _, labels, centres = cv2.kmeans(lab.astype(np.float32), len(colors), None, crit, 5,
                                        cv2.KMEANS_PP_CENTERS)
        bucket_ref = {c: np.mean(np.asarray(pal.buckets[c]), axis=0) for c in colors}
        w = np.array([0.5, 1.0, 1.0], dtype=np.float32)   # hue/chroma matter more than lightness
        best = min(itertools.permutations(range(len(colors))),
                   key=lambda p: sum(float(np.linalg.norm((centres[i] - bucket_ref[c]) * w))
                                     for i, c in zip(p, colors)))
        for i, c in zip(best, colors):
            pal.sand.setdefault(c, []).append(tuple(float(x) for x in centres[i]))


def build_icons(dataset: Path, ann: dict, pal: Palette, v: VisionConfig, out: Path) -> List[str]:
    """Lock / key cells: grey icon templates -> ``out``/<kind>-<colour>-<seq>-r<r>c<c>.png and
    the icon colours -> palette group ``keys``."""
    import cv2

    from cells import icon_lab
    from icons import icon_grey
    from vision import cell_crop, find_lattice

    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    names, colours = [], {}
    for seq, rows in grid_labels(ann).items():
        img = read_shot(dataset, seq)
        lat = find_lattice(img, pal, v)
        for r, row in enumerate(rows):
            for c, label in enumerate(row):
                kind, colour, _, _, _ = parse_label(label)
                if kind not in ("lock", "key"):
                    continue
                crop = cell_crop(img, lat, r, c, v)
                name = f"{kind}-{colour}-{seq}-r{r}c{c}"
                cv2.imwrite(str(out / f"{name}.png"), icon_grey(crop))
                names.append(name)
                lab = icon_lab(crop)
                if lab is not None:
                    colours.setdefault(colour, []).append(lab)
    for colour, labs in sorted(colours.items()):
        pal.keys[colour] = [tuple(float(x) for x in np.median(np.asarray(labs), axis=0))]
    return names


def build_digits(dataset: Path, ann: dict, pal: Palette, v: VisionConfig):
    from digits import DigitReader, group_words, segment
    from vision import crop_rel, find_curtains, find_lattice, lock_digits_roi

    reader = DigitReader()
    problems: List[str] = []

    def learn(text: str, roi: np.ndarray, polarity: str, where: str) -> None:
        glyphs = segment(roi, polarity)
        if len(glyphs) != len(text):
            problems.append(f"{where}: expected {text!r}, segmented {len(glyphs)} glyphs")
            return
        for ch, g in zip(text, glyphs):
            reader.add(ch, g)

    for seq, text in ann["counters"].items():
        learn(text, crop_rel(read_shot(dataset, seq), v.counter), "light", f"counter {seq}")
    for seq, values in ann["fill_percent"].items():
        words = group_words(segment(crop_rel(read_shot(dataset, seq), v.fill_text)))
        if len(words) != len(values):
            problems.append(f"fill {seq}: expected {len(values)} labels, found {len(words)}")
            continue
        for word, val in zip(words, values):
            text = f"{val}%"
            if len(word) == len(text):
                for ch, g in zip(text, word):
                    reader.add(ch, g)
            else:
                problems.append(f"fill {seq}: {text!r} has {len(word)} glyphs")
    for seq, rows in grid_labels(ann).items():
        img = read_shot(dataset, seq)
        lat = find_lattice(img, pal, v)
        for r, row in enumerate(rows):
            for c, lab in enumerate(row):
                kind, _, _, _, counter = parse_label(lab)
                if kind == "ice" and counter is not None:
                    learn(str(counter), lock_digits_roi(img, lat, r, c, v), "light",
                          f"ice {seq} r{r}c{c}")
        for reg in find_curtains(img, lat, pal, v):   # each curtain learns its own label
            counters = {parse_label(rows[r][c])[4] for r, c in reg.cells}
            if len(counters) != 1 or None in counters:
                if counters != {None}:
                    problems.append(f"curtain {seq} {reg.cells}: labels {sorted(map(str, counters))}")
                continue
            bx0, by0, bx1, by1 = reg.box
            sign = img[by0 + (by1 - by0) * 3 // 10: by0 + (by1 - by0) * 7 // 10,
                       bx0 + (bx1 - bx0) // 4: bx1 - (bx1 - bx0) // 4]
            learn(str(counters.pop()), sign, "light", f"curtain {seq}")
    return reader, problems


def build_templates(dataset: Path, ann: dict, v: VisionConfig, out: Path) -> List[str]:
    import cv2

    out.mkdir(parents=True, exist_ok=True)
    names = []
    for name, (seq, rect) in ann["templates"].items():
        img = read_shot(dataset, seq)
        k = v.canonical_width / img.shape[1]
        small = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / f"{name}.png"), crop_rel_px(small, rect))
        names.append(name)
    return names


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.build_palette", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    v = cfg.vision
    dataset = Path(cfg.paths.dataset)
    ann = load_annotations(dataset)

    pal = build_ui(dataset, ann, v)
    build_buckets(dataset, ann, pal, v)
    build_sand(dataset, ann, pal, v)
    icons = build_icons(dataset, ann, pal, v, dataset / "templates" / "icons")
    from tools.add_colour import merge_extra

    extra = merge_extra(pal, dataset / "extra_colours.json")
    pal.save(cfg.paths.palette)
    reader, problems = build_digits(dataset, ann, pal, v)
    reader.save(dataset / "digits.npz")
    names = build_templates(dataset, ann, v, dataset / "templates")

    print(f"palette  : {cfg.paths.palette}")
    print(f"  ui      : {', '.join(sorted(pal.ui))}")
    print(f"  buckets : {', '.join(f'{k}({len(c)})' for k, c in sorted(pal.buckets.items()))}")
    print(f"  sand    : {', '.join(sorted(pal.sand))}")
    print(f"  keys    : {', '.join(sorted(pal.keys))} (icons: {len(icons)} templates)")
    if extra:
        print(f"  extra   : {', '.join(extra)} (dataset/extra_colours.json, tools/add_colour)")
    print(f"digits   : {dataset / 'digits.npz'} — "
          f"{', '.join(f'{k}:{len(t)}' for k, t in sorted(reader.templates.items()))}")
    print(f"templates: {dataset / 'templates'} — {', '.join(names)}")
    for p in problems:
        print(f"warning: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
