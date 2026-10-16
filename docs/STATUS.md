# Sand Loop bot — status / handoff

Read this first, then `docs/GAME_RULES.md` (game mechanics, source of truth) and `PROMPT.md`
(full brief and phase plan). Work phase by phase; after each phase stop and wait for the user's OK.

## Done
| Phase | What | Commit |
|---|---|---|
| 0 | setup, config, permission check | `3b5b8f7` |
| 1 | capture (window + screen rect, hover chrome handled), taps, overlay | `3b5b8f7`, `ba16f6d`, `9cf1d40` |
| 2 | `tools/collect.py`, 32 labeled screenshots in `dataset/screens/` (`LABELS.md`) | `385c523`, `015ab26` |
| 3 | `screens.py`, `vision.py`, `cells.py`, `palette.py`, `digits.py`, `tools/build_palette.py`, `tools/show_annotations.py` | `5d8dd41` |

Branch: `claude/compassionate-dijkstra-osnm1j`. Tests: 154 passed, 1 xfailed (screen 020, see below).

## Key commands
```bash
cd ~/Documents/GitHub/sandloop && source .venv/bin/activate
python -m permissions                    # Screen Recording / Accessibility / Input Monitoring
python -m capture list                   # on-screen windows (find "iPhone Mirroring")
python -m capture info                   # window + detected screen rect + scale
python -m tools.overlay --window         # live view, rel coords under cursor (q/Esc, s = save)
python -m actions tap 0.5 0.8 --dry-run  # tap a relative point (drop --dry-run for a real tap)
python -m tools.collect --max 40         # screenshots -> dataset/screens (ctrl+alt+s snap, ctrl+alt+q quit)
python -m tools.build_palette            # rebuild palette.json, digits.npz, templates/ from annotations.json
python -m tools.show_annotations         # draw annotated grids -> debug/annotations/
python -m vision dataset/screens/*-043-auto.png   # read a gameplay frame -> debug/vision-*.png
python -m screens dataset/screens/*.png  # classify screen types
python -m pytest -q                      # all tests, no live game needed
```
Global hotkeys: `ctrl+alt+p` pause, `ctrl+alt+q` quit. Cursor in a screen corner = failsafe stop.

## Known decisions
- Playability = **path rule** via BFS (`cells.path_playable`): a bucket is playable iff a
  4-neighbour path of empty cells connects it to the exit (front row at the conveyor side).
- **Tap only when BFS and the visual face-up state agree**; otherwise log a warning, don't tap
  (`GameView.tappable()`).
- Every cell: empty / bucket / known lock / UNKNOWN. UNKNOWN is never tapped; frame + crop go to
  `dataset/unknown/` (`vision.save_unknown`). Lock types are data-driven (`cells.CELL_LOCKS`,
  region detectors such as `vision.find_curtains`).
- **Towers**: only the top stage is tappable; the next stage unlocks when the top bucket is 100%.
- **Curtains**: size varies, detected from the gold frame; buckets underneath (colours unknown
  until it opens), then regular buckets under the path rule. Opens after N buckets reach 100%.
- **Ice N**: opens after N more buckets are tapped.
- Outside gameplay: **revive → Quit**, then **Try Again** on LEVEL FAILED; **ads/popups → only ✕**
  (wait for it; never Get App / purchases / Activate; close App Store sheets); **no boosters**;
  **stop at 0 lives**.
- No hardcoded RGB: colours come from `dataset/palette.json`; tunables live in `config.yaml`.
  The k/5 counter is authoritative for conveyor fullness (edge slots may be `uncertain`).

## Open questions / known gaps
- **Win screen (level_complete)**: not captured yet — user will grab one with `ctrl+alt+s`.
- **Screen 020** (ad end card without Get App / mute / ✕): no anchor, classified `other`
  (xfail in tests). Needs more ad end-card shots, ideally when the ✕ appears, to add a generic
  ad anchor.
- **Shot 010, cell r4c4 (0-based)**: blue bucket looks face-down although the path rule says
  playable. Asked the user whether flips are delayed or the rule has a nuance. Bot doesn't tap it.
- **Face-up tower** (shot 031 r1c2) reads face-down (one sample); more tower shots needed.
- Popup ✕ template threshold is tight (real 0.96–1.00 vs false up to 0.93).

## Phase 4 — next
- `state.py`: dataclass model of the level state built from `vision.GameView` (grid cells,
  playable/tappable buckets, locks with counters, conveyor k + slot colours/fill, remaining sand
  per colour and exposed bottom colours).
- `strategy.py` (greedy): choose the next bucket per `GAME_RULES.md` "Strategy implications"
  (release only colours whose sand is exposed and sufficient, prefer moves that open paths /
  unlock locks, keep a conveyor slot usable; never tap UNKNOWN or disagreeing cells).
- `--dry-run` in the overlay: draw detections and the chosen move, never click.
- Tests on the saved screenshots; then stop and wait for the user's OK.
