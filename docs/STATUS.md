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
| 4 | `state.py`, `strategy.py` (greedy), `tools/overlay.py --dry-run`, `config.yaml` `strategy:` | `6ff1c26`, `9f2b4e7`, `774d07e`, + misread safeguards |

Branch: `claude/compassionate-dijkstra-osnm1j`. Tests: 507 passed, 1 xfailed (screen 020, see below).

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
python -m strategy dataset/screens/*.png # greedy decision per shot -> debug/strategy-*.png
python -m tools.overlay --dry-run        # live: detections + chosen move, never clicks
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
- **Towers**: tapping sends the whole tower to the belt (cell empty at once, one belt slot); its
  stages collect in order on the belt (next stage starts when the current one is 100%).
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
- Popup ✕ template threshold is tight (real 0.96–1.00 vs false up to 0.93).
- **Shot 035**: darkred r2c3, darkred r4c1, cream r4c3 look face-up although BFS says closed
  (they were face-up in 033), and an orange bucket sits in r2c2 that was empty in 033. Do
  buckets stay face-up once opened? Can buckets move/appear in empty cells? Asked the user.
- **Ice**: what is under an ice block once it opens (a bucket? empty?) is unknown, so ice cells
  are not counted as pending buckets in the capacity estimate.
- Screen classification costs ~0.7 s per frame (template matching at 8 scales); the overlay runs
  the analysis on a worker thread. Phase 5 may want a cheap gameplay pre-check.

## Phase 4 — greedy strategy (done, awaiting OK)
- `state.LevelState.from_view(GameView)`: grid, tappable, locks (ice / curtain regions with
  counters), conveyor (k/5 authoritative; uncertain edge slots dropped, empty at k=0), sand and
  exposed shares, pending buckets (grid + one hidden per curtain cell + conveyor need) and a
  capacity estimate = remaining sand / pending buckets. `after_tap()` / `opened_by()`: one-tap
  lookahead (cell empties, ice counters count down; towers: see "L183 towers" below).
- `strategy.choose()`: hard rules (tappable only; free slot; colour exposed >= `min_exposed`,
  or >= `last_slot_min_exposed` for the last free slot; enough sand left), then score =
  exposed share + buckets/lock cells opened. Weights/thresholds in `config.yaml` `strategy:`.
- `strategy.Analyzer`: frame -> screen type -> GameView -> LevelState -> Decision (+ tap point).
- Fix after the live dry-run: face up/down is judged on the top stage only (centre vs the rim =
  85th percentile of V in the top part). A tall tower's opening is drawn above the cell, so the
  old median reference read it face-down and L181 start (030/041) had no tappable bucket.

- Live dry-run feedback (L181, shot 052 `-live`): it chose red while a red bucket at 49% on the
  belt was starving. Now: "exposed" = the **contact layer** (lowest ~10 px above the picture's
  real bottom edge, found by walking past `vision.picture`; the old band measured the pile's
  slope). **Demand-aware**: avail = exposed - `belt_demand_exposure` x (1 - fill) per
  same-colour belt bucket. **Stalled**: `state.BeltTracker` keeps per-colour fill across frames;
  no growth for `stall_s` = colour rejected. New colours get a small bonus and win ties. Each
  candidate's reason shows `exp / belt / avail / stalled / new`.
- Overlay reports: `ctrl+alt+s` (global, confirmed live) or `s` (overlay focused) ->
  `debug/report-<time>.png` (now), `-analysed.png` (frame the decision is about), `-annotated.png`,
  `.txt` (decision + every candidate). Plain `s` over iPhone Mirroring goes to the phone.
- Live: some frames fail vision intermittently ("grid panel background not found", "counter
  not read"); the same position re-analysed is fine. Must be fixed before phase 5:
  - every failed gameplay read (step `grid` or `counter`) -> `debug/vision_fail/<time>-<step>.png`
    + `fails.jsonl` (failure, frame diff, running counts), capped by `vision_fail_max_saves`;
  - `strategy.DecisionGate`: a move counts only after `confirm_frames` (2) consecutive analyses
    chose the same cell + colour with no failed read in between; phase 5 taps only confirmed
    moves and calls `reset()` after each tap;
  - overlay status line: `pending 1/2` / `CONFIRMED 2/2` and `read fails f/n (p%)`.
  - **Collected so far (2026-10-16): nothing.** `debug/vision_fail/` is empty — 0 failed
    reads, so 0 grid / 0 counter, no pattern yet. The logging overlay ran ~2 h, but the game sat
    on one idle L181 position (every analysis confirmed `tap r3c0 orange`, no failure), then
    only non-gameplay screens. The earlier failures (before logging existed, seen only in the
    console: "grid panel background not found", "conveyor counter not read") happened while
    the user was actively playing — suspect belt/sand animation or transitions, unconfirmed.
  - **Root cause found offline (2026-10-16), fixed, awaiting live confirmation.** Both readers
    were fragile to a crop a few px off (pt/px rounding, window edge, hover-chrome frame):
    a 3 px foreign strip at the left edge -> "grid panel background not found" on every frame
    (the panel test used the outermost 5 % column strip); a 4-10 px vertical offset -> "counter
    not read" (glyphs touching the tight zone's border were dropped). Fix: the grid finder ignores
    `vision.edge_margin` at both edges and accepts the left or right strip; `vision.read_counter`
    slides the tight zone by up to `vision.counter_search`. Tests `*_survives_misaligned_crop`
    (+-3/8 px x, +-4/10 px y). Still failing by design: a dimmed frame (popup fade, V x0.7).
  - Previously NEXT (before phase 5), now = verification: run `python -m tools.overlay --dry-run` while actually playing
    (belt moving, level end / retry, ads), then inspect `debug/vision_fail/fails.jsonl`
    (step, failure, frame diff) and the saved frames, find the cause, fix it.
- Env note: `.venv` is Python 3.9.6 (Xcode); CLAUDE.md says 3.11+. Works (future annotations).

## L183 (2026-10-16): new colours, locks / keys, curtains — done, awaiting OK
- Live dry-run with the edge/counter fix: 1 failed read in 154 (a frame grabbed while the window
  was being moved: stale geometry, frame cut off — failing it is correct).
- Dataset: shot 053 (L183 start, from the overlay report) annotated in `annotations.json`
  (new labels `lock:<colour>`, `key:<colour>`; `sand_colors` for colours hidden under curtains /
  locks, e.g. red). Palette rebuilt: buckets + sand `green`, `darkgreen`; new group `keys`
  (`yellow`, `red`); icon templates in `dataset/templates/icons/`.
- Shades: green vs darkgreen buckets are the closest pair (14.8 LAB between nearest looks,
  max_dist 22: nearest-colour still separates them on every annotated cell); sand 34 apart.
  The sky is the existing `cyan` sand.
- Curtains: posts now end at their first gap (the top curtain used to swallow the two below);
  L183 reads 10 / 15 / 20; build_palette learns each curtain's own counter.
- Towers seen from above: a disc on a dome = face-down tower `<disc>/<dome>-down` (rule refined
  in the next section).
- Locks / keys: `vision.IconMatcher` (grey shape templates, colour-independent: other-colour
  icons 0.82-0.85, other cells <= 0.62, threshold 0.75), colour from palette `keys`; cell types
  in `cells.CELL_LOCKS` (`lock` unlock "key", `key` unlock "path"). `state.after_tap` releases
  keys the tap connects (chains), empties their cells and opens the lock of their colour
  (`LevelState.unlocked`); keys already connected give no credit. Strategy: +
  `weight_key_unlock` per bucket under the opened lock ("opens lock yellow with 1 bucket(s)").
- Unknown colours: `GameView.unknown_colours` (bucket / lock / key cells + `vision.unknown_sand`
  clusters >= `unknown_sand_min`), logged, magenta in the overlay, saved once per run to
  `dataset/unknown_colours/` (+ `unknown_colours.jsonl`); `python -m tools.add_colour <name>
  <files>` -> `dataset/extra_colours.json` (merged by build_palette) + rebuild. The Analyzer now
  also saves UNKNOWN cells to `dataset/unknown/` (was never called live).
- L183 start decision: tap r0c2 darkgreen (contact layer: darkgreen 61 %, green 39 %) —
  confirmed right by the user.
- User answers: a flown key's cell is empty (as modelled). Several same-colour locks: not seen
  yet (bot credits no unlock then). Open: multi-cell locks, next live check of L183.

## L183 live fixes (2026-10-16) — done, awaiting OK
- Regression 054 (live): a face-up darkgreen bucket on a cyan tower stage (r2c5, next to the
  yellow key) read face-down -> "no tappable bucket". Cause: the disc rule (added for L183) took
  the open bucket's dark inside for a disc because its hue differs from the stage below. Now a
  disc = centre and the **ring around it** are different palette colours >= `disc_min_dist`
  apart (open bucket: ring = its own rim, <= 5 apart; disc on a dome: >= 76). Expected move
  r2c5 darkgreen: tested, with key-unlock credit for the yellow key (see "L183 towers").
- capture: `check_aspect` rejects a detected screen whose width/height is not the iPhone's
  (`screen.aspect` 0.4624 +- `aspect_tolerance` 2 %); `grab_content` raises
  `FrameSizeMismatch` for a crop of the wrong size (window moved / partly off-screen; live:
  502 px wide frames), `WindowTracker.grab` re-resolves once and retries. Not covered: a
  full-size frame with the content shifted mid-animation (the reader rejects it; 2-frame
  confirmation guards taps).

## L183 towers: the whole tower goes to the belt (user correction) — done, awaiting OK
- The old rule ("next stage stays in the cell, locked until the top is 100%") was wrong: the
  whole tower leaves, its cell empties at once, one belt slot; the stages fill in order ON the
  belt. L181 shots agree (030 -> 031: r0c2 empty, cream/red tower on the belt).
- `state.after_tap`: tower -> EMPTY cell + `ConveyorBucket(top, 0, stages=(body,))`;
  `ConveyorBucket.need_for` / `conveyor_need` count later stages as demand of their colour,
  `pending_buckets` too. Regression 054: tap r2c5 now gets "opens lock yellow with 1 bucket(s)".
- Strategy: a tower scores by its top colour; minus `weight_tower_short_stage` (0.5) per later
  stage whose colour fails the sufficiency test (it would block the belt slot); shown as
  "tower then cyan (sand short)".
- Belt reader: `Slot.body` = next stage, read as the base colour under the collecting bucket
  (brighter half of the lowest 30 % of the belt zone, middle of the run); `annotations.json`
  `belts` (8 shots incl. cream/red 34 % in 033, red/orange 7 % in 032), tested. Known miss: a
  tower still dropping onto the belt (031) has its base below the zone and reads one-colour
  for that frame. Only the next stage is visible (deeper stages unknown).
- Open (phase 5): remember the stages of towers the bot tapped itself, to cover the belt view.

## Unlock value + path planning (2026-10-16) — done, awaiting OK
- User request: score what a move opens by colour usefulness, plan paths to useful hidden
  buckets / keys (e.g. L183: open the way to a key). Rules in GAME_RULES.md "Unlock value and
  paths"; code `planner.py` (usefulness, `opened_value`, Dijkstra `shortest_path`, `replay`
  safety check, `plan`); `strategy.evaluate` = the hard rules (shared with the path check),
  `choose` adds the unlock value and the path bonus; `Decision.plans` (best first, rejected
  with reason) in the overlay text, `.txt` report ("paths:") and `python -m strategy`.
- Vision: `GameView.lower` / `LevelState.lower` = sand share per colour in the lower
  `vision.lower_band` (0.3) of the picture zone.
- `weight_open_bucket` 0.15 -> 0.05 (colour worth now comes from `weight_unlock_value`); all
  saved-shot decisions unchanged. Tests `tests/test_planner.py` (greedy-wrong / path-right grid,
  deadlock and starving-step rejections, key target).
- Not live-tested yet. Weights are first guesses.

## Hint arrows (2026-10-16, live L183 shot 055) — done, awaiting OK
- Live: "wait: no tappable bucket" with face-up creams visible. Cause: the yellow hint arrows
  (shown at 4/5 with buckets not full) are not panel-coloured, so `find_lattice` joined the
  grid to the booster buttons -> 8x8 lattice, every cell misplaced, a phantom orange row, every
  face-up bucket "disagrees" with the path rule. Not the last-slot rule.
- Fix: `hints.py` (`HintArrows` masked template match on the grid zone at half resolution,
  ~40 ms; `occlusion_mask`, `clean` = inpaint, `CellMemory` = last clean cell readings per
  lattice); `find_lattice(occluded=)` ignores arrow pixels; `read_grid` reads cells through
  the inpainted mask, falls back to memory / UNKNOWN-occluded (`Cell.occluded`: never tapped,
  not in `unknown_cells()`); `GameView.hints` drawn as grey boxes. Analyzer keeps the memory.
- 055 now reads like 054's lattice; bottom row cyan-down / orange-down read through the mask.
  Decision on 055: wait, all three face-up buckets fail the **last-slot rule** (4/5 -> avail
  >= 25 %): cream r3c0 / r4c5 0 % exposed, darkgreen r4c0 24.6 %.
- `choose` now lists face-up buckets that are not candidates with the reason (path rule
  disagreement / hidden by a hint arrow); exposure rejections show one decimal.
- Tests `tests/test_hints.py` (arrows only on 055, lattice = 054's, labels, memory, last-slot
  decision) + a reason test. Shot 055 is not in `annotations.json` (build_palette's lattice
  has no arrow mask).
- `dataset/unknown_colours/20261016-181159-393-bucket-r3c1.png` (untracked) is a misread
  panel-background crop from the arrow frames before the fix: junk, can be deleted.

## Open-bucket colour from the rim; vision.py split (2026-10-16) — done, awaiting OK
- User: 055 r4c0 is an open LIGHT green bucket; the bot read `darkgreen/green-up` (the shaded
  inside -> darkgreen top, the light rim -> a fake green stage) and rejected it at the last slot
  (24.6 % darkgreen exposed). Light green is 47 % of the contact layer.
- Fix: `CellFeatures.rim_lab` = top-stage object pixels at V >= `vision.rim_percentile` (75;
  85+ pulls darkgreen rims toward green); `classify_cell` reads face-up buckets from it
  (`Cell.seen_lab` / `seen_from` = rim | top | disc). `build_palette` learns an `up-rim` look
  per colour from annotated open buckets (palette rebuilt: only that look added). All 33
  annotated open buckets classify right.
- Greens on 053 / 054 / 055 re-checked: every one matches the annotation; nearest-vs-other
  margins >= 10.9 (open darkgreen rims), 055 r4c0 green 2.4 vs darkgreen 18.0. 055 is not in
  `annotations.json`; its expected grid lives in `tests/test_hints.py` (r4c0 = `green-up`).
- 055 decision: **tap r4c0 green** (score 0.82; opens r5c0 darkgreen (useful)).
- Debug: `GameView.colour_notes` (vision / strategy CLI, overlay report "colour close calls"):
  every bucket of a `vision.colour_watch` pair (green / darkgreen) and any bucket whose
  runner-up colour is within `color_max_dist`, with both distances, `CLOSE` under
  `tower_min_margin`.
- vision.py split (406 -> 433 lines with the notes): `belt.py` (counter, conveyor slots),
  `sand.py` (sand shares, contact layer, lower part), `zones.py` (`crop_rel`).

## Phase 5 — next
- `bot.py`: full loop on one level (stable frame -> Analyzer -> tap via actions.Tapper with all
  safety checks), log every move (frame + decision) to `debug/`.
