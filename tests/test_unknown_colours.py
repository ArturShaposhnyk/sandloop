"""Colours missing from the palette are reported and saved, never silently dropped, and
tools/add_colour turns the saved crops back into palette colours (L183 grass greens)."""
import copy
import json
from pathlib import Path

import pytest

from cells import Kind
from config import load_config
from digits import DigitReader
from palette import Palette
from tools.add_colour import add_extra, colours_from_crop, group_of, merge_extra
from tools.build_palette import read_shot
from icons import IconMatcher
from unknown_colours import colour_key, save_unknown_colours, unknown_sand
from vision import read_gameplay

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
GREENS = ("green", "darkgreen")
LABELS = {l.split("|")[1].strip(): l.split("|")[2].strip()
          for l in (DATASET / "screens" / "LABELS.md").read_text().splitlines()
          if l.startswith("| 2026")}
GAMEPLAY = sorted(name.split("-")[3] for name, kind in LABELS.items() if kind == "gameplay")


@pytest.fixture(scope="module")
def ctx():
    cfg = load_config(ROOT / "config.yaml")
    return (cfg.vision, Palette.load(DATASET / "palette.json"),
            DigitReader.load(DATASET / "digits.npz"),
            IconMatcher.load(DATASET / "templates" / "icons"))


def without_greens(pal: Palette) -> Palette:
    p = copy.deepcopy(pal)
    for group in (p.buckets, p.sand):
        for name in GREENS:
            group.pop(name, None)
    return p


@pytest.mark.parametrize("seq", GAMEPLAY)
def test_known_palette_reports_no_unknown_colour(seq, ctx):
    v, pal, rd, icons = ctx
    gv = read_gameplay(read_shot(DATASET, seq), pal, v, rd, icons=icons, log=lambda s: None)
    assert gv.unknown_colours == []


def test_missing_greens_are_reported_logged_and_saved(ctx, tmp_path):
    v, pal, rd, icons = ctx
    img = read_shot(DATASET, "053")
    logs = []
    gv = read_gameplay(img, without_greens(pal), v, rd, icons=icons, log=logs.append)
    cells = {u.cell for u in gv.unknown_colours if u.kind == "bucket"}
    # the green / darkgreen buckets of the front row and r4c0, r4c4, r5c0, r5c6
    assert {(0, 2), (0, 3), (0, 4), (4, 0), (4, 4), (5, 0), (5, 6)} <= cells
    for r, c in cells:
        assert gv.grid[r][c].kind is Kind.UNKNOWN          # never tapped
    sand = [u for u in gv.unknown_colours if u.kind == "sand"]
    assert len(sand) == 2 and all(u.share > 0.1 for u in sand)   # grass: light + dark green
    assert any("unknown sand colour" in line for line in logs)

    seen: set = set()
    saved = save_unknown_colours(img, gv, v, tmp_path, seen, log=lambda s: None)
    assert len(saved) == len(gv.unknown_colours) - _duplicates(gv)
    assert save_unknown_colours(img, gv, v, tmp_path, seen, log=lambda s: None) == []  # once
    recs = [json.loads(l) for l in (tmp_path / "unknown_colours.jsonl").read_text().splitlines()]
    assert {r["kind"] for r in recs} == {"bucket", "sand"}
    assert all((tmp_path / r["file"]).exists() for r in recs)


def _duplicates(gv) -> int:
    keys = [colour_key(u) for u in gv.unknown_colours]
    return len(keys) - len(set(keys))


def test_add_colour_from_saved_crops_restores_the_greens(ctx, tmp_path):
    import cv2

    v, pal, rd, icons = ctx
    img = read_shot(DATASET, "053")
    bare = without_greens(pal)
    gv = read_gameplay(img, bare, v, rd, icons=icons, log=lambda s: None)
    save_unknown_colours(img, gv, v, tmp_path, log=lambda s: None)
    files = {p.name: p for p in tmp_path.glob("*.png")}
    # the user names them: front-row r0c3 / light sand = green, r0c2 / dark sand = darkgreen
    light_sand = max((u for u in gv.unknown_colours if u.kind == "sand"), key=lambda u: u.lab[0])
    dark_sand = min((u for u in gv.unknown_colours if u.kind == "sand"), key=lambda u: u.lab[0])
    pick = {
        "green": [n for n in files if n.endswith("-bucket-r0c3.png")
                  or n.endswith(f"-sand-{'_'.join(f'{x:.0f}' for x in light_sand.lab)}.png")],
        "darkgreen": [n for n in files if n.endswith("-bucket-r0c2.png")
                      or n.endswith(f"-sand-{'_'.join(f'{x:.0f}' for x in dark_sand.lab)}.png")],
    }
    extra = tmp_path / "extra_colours.json"
    for name, names in pick.items():
        assert len(names) == 2, names
        for n in names:
            group = group_of(files[n])
            add_extra(extra, group, name, colours_from_crop(cv2.imread(str(files[n])), group,
                                                            bare, v))
    assert sorted(merge_extra(bare, extra)) == ["buckets/darkgreen", "buckets/green",
                                                "sand/darkgreen", "sand/green"]
    again = read_gameplay(img, bare, v, rd, icons=icons, log=lambda s: None)
    assert again.unknown_colours == []
    assert [again.grid[0][c].label() for c in (2, 3, 4)] == [
        "darkgreen-up", "green-up", "darkgreen-up"]
    assert again.sand["green"] > 0.1 and again.sand["darkgreen"] > 0.1


def test_group_from_file_name():
    assert group_of(Path("20261016-170000-123-bucket-r0c3.png")) == "buckets"
    assert group_of(Path("20261016-170000-123-sand-89_-61_61.png")) == "sand"
    assert group_of(Path("20261016-170000-123-key-r2c6.png")) == "keys"
    assert group_of(Path("crop.png")) is None


def test_scattered_leftovers_are_not_a_colour(ctx):
    v, pal, _, _ = ctx
    assert unknown_sand(read_shot(DATASET, "011"), pal, v) == []   # late frame, most noise
