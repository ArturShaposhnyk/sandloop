# Sand Loop bot — status / handoff

Read this first, then `docs/GAME_RULES.md` (game mechanics, source of truth) and `PROMPT.md`
(full brief and phase plan). Work phase by phase; after each phase stop and wait for the user's
OK. Branch: `claude/compassionate-dijkstra-osnm1j` (pushed).

## Done (phases 0-6)
| Phase | What | Commits |
|---|---|---|
| 0 | setup, config, permission check | `3b5b8f7` |
| 1 | capture (window + screen rect, hover chrome handled), taps, overlay | `3b5b8f7`, `ba16f6d`, `9cf1d40` |
| 2 | `tools/collect.py`, labeled screenshots in `dataset/screens/` (`LABELS.md`) | `385c523`, `015ab26` |
| 3 | `screens.py`, `vision.py`, `cells.py`, `palette.py`, `digits.py`, `tools/build_palette.py` | `5d8dd41` |
| 4 | `state.py`, `strategy.py` (greedy + path planning), `tools/overlay.py --dry-run` | `6ff1c26`, `9f2b4e7`, `774d07e`, ... |
| 5 | `bot.py` plays a level with real taps | `4b1883a`, `4f297f2`, `00daa0e` |
| 6 | screens outside gameplay (`flow.py`, `levels.py`, `adclose.py`, `hud.py`), `--levels N`, session limits | `a07c297`, `c1b1423` |
| fixes | test clock (`85e5960`); L187 lattice / colours / locks (`76ea470`, `5750651`); unlock rule + this handoff (`c9e26e2`) | |

Tests: `python -m pytest -q` - all pass except 1 documented strict xfail (open tower, below).
The shared test Analyzers use a frozen fake clock, so results never depend on machine load.

## Where it stands (2026-10-20)
- **Live wins: L186** by itself (run `20261019-171033`, 21 taps, 118 s). **L187 (HARD)**: one
  live run (`20261020-122638`) got STUCK at 3/5 on a lattice misread - fixed offline (below),
  not live-tested again yet. No full L187 win yet.
- **Phase 6 is not exercised live yet**: the only `--levels` run so far was L187 with
  `--levels 1` (stopped by the stuck lattice / window loss), so no NEXT / PLAY / Quit /
  Try Again / ad ✕ tap has happened live. Revive popup ✕ = Quit (user-confirmed).
- **L187 grid fix**: lane walls every two columns made 2x the cell pitch win the column
  autocorrelation, and as the lanes emptied the bot switched to 4x4 mid-level (fake towers,
  phantom colour LAB 74/19/0, STUCK). Now: walls must sit on cell boundaries
  (`vision.pitch_wall_align`), `LatticeLock` refuses the same outline with another cell size,
  and a cell with background between two objects is UNKNOWN (`vision.split_bg_max`).
- **L187 colours / icons**: yellow + purple (sand via `tools.add_colour` ->
  `dataset/extra_colours.json`, buckets from annotated shot 064); blue locks / keys; icons are
  matched on the cell's own rect (`vision.icon_crop`).
- **Unlock rule (user, 2026-10-20, `c9e26e2`)**: a bucket under a just-opened lock is tapped
  at once only with a confident colour (not the lock's colour, nearest <=
  `vision.unlock_confident_dist`, runner-up >= `unlock_confident_margin` further); else it
  waits until the padlock is gone (`vision.unlock_wait_s` 2 s). `vision.unlock_unsure`,
  `GameView.unsure`, logged as "just unlocked, colour not sure yet". The old frames read the
  padlock's colour (00285 "orange" was only luck before yellow existed).

## NEXT
1. Live: `python -m bot --levels 1` on L187 - expect no `lattice: new grid geometry` /
   `same outline` lines; then `--levels 2`-`3` to exercise phase 6 (watch it).
2. Answers on phase-5 issues A / B, then fixes A-D (see "Phase 5 state" below).
3. Separate fix later (user): the open-tower gap.

## Open gaps / questions
- **Open tower in a tall lane** (strict xfail `test_open_tower_keeps_its_lower_stage`): L187
  r1c0 orange-on-cream reads `orange-up`, the cream stage is drawn below the bucket crop. The
  belt reader sees the stage once it is on the belt; the lookahead under-counts it.
- **Two touching buckets in one cell** score ~0.1 on the split check (not caught); only the
  lattice checks prevent such cells.
- **Chest-opening flash now reads yellow** (was an unknown colour, LAB ~98/-18/68: 11-15 from
  palette yellow). Seen only inside the chest-settle window (no decisions for
  `strategy.chest_settle_s` after a chest opens; the flash lasted <= ~1 s) - watch it on a
  level with chests AND yellow sand.
- Yellow / purple: only face-down looks learned (no open bucket of them seen yet).
- Phase 6 not covered by shots: lives 0, a real ad ✕ in a corner (only the App Store sheet's),
  "no lives" popups, ad end cards outside `flow.close_zones`.
- A window dragged partly off-screen can still give full-size frames with shifted content
  (run 122638 frame 00058); the bot ended on "window lost" there.
- Phase-5 issues A-D (exposure-0 opener, negative-score bursts, false no-effect, chest flash
  logging) - waiting for the user's answers on A / B.
- Older questions: shot 010 r4c4 face-down although playable; shot 035 buckets face-up although
  closed; what is under ice; L186 chest landing cells; popup ✕ threshold tight (0.96 vs 0.93);
  screen classification ~0.7 s/frame.

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
python -m bot --dry-run                  # phase 5 loop, logs to debug/runs/, never clicks
python -m bot --max-taps 3               # real taps, stop after 3
python -m bot --levels 3                 # phase 6: 3 levels in a row (NEXT / PLAY / Quit / Try Again / ad ✕)
python -m tools.analyze_run              # post-mortem of the latest run
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
- **Key locks**: buckets under a lock are playable as soon as it opens; for
  `vision.unlock_trust_s` the path rule alone decides there (padlock animation reads face-down);
  tapped at once only with a confident colour, else after `vision.unlock_wait_s`.
- **Stalled colour** = lost slot: rejected while another belt bucket collects, else a penalised
  last resort (`weight_stalled_colour`; path steps cost `path_cost_stalled`).
- **Chests / mystery / walls** (L186): `chestN` and `mystery` are single-cell locks
  (`cells.CELL_LOCKS`, palette ui `chest_wood` / `mystery_grey`); walls are `x`. Opening a chest
  is priced by `planner.scatter_risk`; after one opens the Analyzer waits `chest_settle_s` and
  re-reads the whole grid.
- **Grid geometry** is fixed per level (`hints.LatticeLock`, enabled by
  `Analyzer.reset_level()` in bot.py / overlay; off for unrelated screenshots). The column
  pitch must fit the walls inside the grid; the same outline with another cell size is refused.
- Outside gameplay: `flow.py` picks the only allowed button (NEXT / PLAY / Quit / Try Again /
  ✕); never inside Keep Playing / Watch Ad / Get App / App Store boxes; every tap logged.
- Frame -> decision lives in `analyzer.py` (Analysis, Analyzer, DecisionGate); `strategy.py`
  is the greedy choice only.

# History (newest first)

## L187 live run 20261020-122638 (2026-10-20): STUCK at 3/5 — fixed, awaiting live check
- Started right (7x7), 10 good taps; at 29 s the lattice turned 4x4 and LatticeLock took it
  after 3 frames ("new grid geometry 4x4 (was 7x7)"): every cell held 2x2 buckets -> fake towers
  ("cream/blue-up", "purple/cream-up"), the unknown colour, STUCK. Cause: the half-pitch
  autocorrelation peak comes from the buckets in the lanes and faded as they left (0.50 -> 0.38
  vs the walls' 2x peak 0.52), below `pitch_harmonic_ratio`. The run ended "window lost" when
  the window was dragged off-screen (frame 00058), not by the bot.
- Fixes: (1) `vision.wall_alignment`: the grid's inner walls (lanes, notches) must sit on cell
  boundaries (`vision.pitch_wall_align` 0.8), else 1/2, 1/3 of the pitch is tried -> every frame
  of the run 7x7. (2) `LatticeLock`: the same outline with another cell size is a misread,
  refused for any number of frames (a new level comes after non-gameplay screens, which reset
  the lock); its messages now show the cell size. (3) `cells`: a background row through a
  cell's middle with object pixels above AND below it (`vision.split_bg_max` 0.15) = two
  objects -> UNKNOWN, never a bucket / tower. Annotated buckets <= 0.042, fading padlocks
  <= 0.066; of the run's 16 merged 4x4 cells 10 score >= 0.17, but two touching buckets in a
  lane (the run's "cream/blue-up", "cream/purple-up") score ~0.1 and are NOT caught: pixels
  alone cannot split them reliably, the two lattice checks are the real guard.
- LAB 74/19/0 (r3c2) was NOT a colour: the 4x4 cell held an orange bucket bottom, a purple edge
  and the blue key (median of the mix). Not added. No other unknown colour in the run.
- Known gap (strict xfail): an OPEN tower in a tall lane (r1c0, orange on cream) reads
  `orange-up` - the cream stage is drawn below the bucket crop. The belt reader still sees the
  stage once it is on the belt; the lookahead under-counts its demand.
- Tests: frames 00030 / 00041 in `tests/data/run-20261020-122638/`, `tests/test_run_20261020.py`.

## L187 dry run 20261020-120601 (2026-10-20) — fixed, awaiting the user's live check
- Root cause of the "cream-up looks face-up but the path rule says closed" warnings: the
  **lattice**, not colours. L187 has walls every two columns; the column autocorrelation peak
  at 2x the pitch (112 px, 0.511) beat the real one (56 px, 0.502) -> a 4x4 lattice, every cell
  spanning several buckets. `vision._fundamental`: a peak at 1/2 or 1/3 of the best that is
  >= `vision.pitch_harmonic_ratio` (0.8) x as strong is the pitch. Now 7x7; all other shots
  unchanged.
- Colours: sand yellow + purple via `tools.add_colour` (-> `dataset/extra_colours.json`);
  yellow / purple **bucket** looks from shot **064** (annotated) - the run's bucket crop was cut
  by the wrong lattice (two yellows + blue/orange neighbours). Yellow vs cream: >= 54 LAB apart
  on every annotated bucket. Only face-down yellow / purple looks so far (no open one seen).
- Locks / keys: blue is a new key colour (palette keys blue / red / yellow). The yellow key was
  read as orange-down: icons were matched on the bucket crop shifted up, i.e. the rim of the
  bucket above + a cut-off key; with L187 templates empty cells then matched a key at 0.80.
  `vision.icon_crop` (cell's own rect, `vision.icon_inset` 0.05): icons with their own template
  left out >= 0.88, other cells <= 0.65 (threshold 0.75). L183 reads unchanged.
- Start decision on 064: tap r0c3 orange (47 % exposed), path to r2c3 orange.
- Tests: `tests/test_run_20261020.py` + 064 in every annotated-grid test of `test_vision.py`.
- **Open question (found through yellow)**: right after a key opens a lock, the cell still shows
  the padlock body (run 20261017 frames 00285 r2c1, 00304 r3c5) and is read as a bucket of the
  LOCK's colour. 00285 used to read "orange" only because yellow was missing (nearest colour);
  now it reads yellow. RESOLVED 2026-10-20 (user): tap at once only with a confident colour,
  else wait until the padlock is gone (`vision.unlock_unsure`, test
  `test_opened_lock_waits_for_a_confident_colour`; the xfail is gone).

## Phase 6 — level after level (2026-10-19) — done offline, awaiting the user's live check
- Screens: win -> **NEXT** (template `next`; `watch_ad` is forbidden), menu -> red ✕ of a popup
  else **PLAY**, revive -> **Quit** else the popup's red ✕ (038; same as Quit -> LEVEL FAILED,
  user-confirmed 2026-10-20),
  fail -> **Try Again**, ads -> "Take a break" wait / generic ✕ (`adclose.py`, X-shaped glyph
  in corner zones; 018's App Store ✕ found, no hits on gameplay / menu / win shots), unknown ->
  wait (✕ search too within `flow.ad_context_s` after an ad). `screens.FORBIDDEN`: no tap inside
  Keep Playing / Watch Ad / Get App / App Store badge boxes.
- Taps need `flow.confirm_frames` agreeing frames, `tap_cooldown_s` apart; stops: same button
  `same_button_max` x without effect, `max_taps_between_levels`, screen other > `unknown_max_s`,
  ad without ✕ > `ad_max_s`, known screen with nothing allowed > `stuck_max_s`, lives 0
  (`hud.read_lives`: digit in the heart, palette ui `heart_red`; PLAY / Try Again need lives
  >= 1). Every stop saves the frame; every screen tap logs raw + annotated frame
  (`screen_tap` event; `tools.analyze_run` lists them and level results).
- Session (`config.yaml session:`): `--levels N` (default 1 = stop on the end screen, like
  phase 5) capped by `max_levels`, `max_total_minutes`, `max_consecutive_fails`; `bot.max_run_s`
  is now per level. Revive / fail = one lost level.
- Dataset: win shots renamed 001-005 -> **059-063** (their seqs clashed with L180 shots), labeled
  `level_complete`; new templates `level_completed`, `next`, `watch_ad`, `status_icons` (iOS
  status bar = ad: fixes 020, xfail removed); ui colour `heart_red`. Palette otherwise unchanged.
- Not covered by shots: lives 0, a real ad ✕ in the corner (only the App Store sheet's),
  "no lives" popups, ad end cards outside `close_zones`.
- Fixes A-D from phase 5 (below) are still open; E (level_complete template) is done.

## Phase 5 state (2026-10-19)
- **Phase 5 (`bot.py`) is done: it won L186 live by itself** (run
  `debug/runs/20261019-171033`: 21 taps, 118 s, 0 read failures, 0 no-effect taps; result
  `win`, inferred from sand 0 % then a non-gameplay screen). Live runs that day, all L186:
  `162703` (9 good taps, then the phantom-7th-row freeze, fixed with `hints.LatticeLock`),
  `170512` / `170812` / `170918` (`--max-taps 10`, no freeze), then `171033` (the win).
- Win screen captured: "Well Done! / Level completed!" with NEXT and Watch Ad (shots
  `dataset/screens/20261019-171450-*` .. `20261019-171515-*`, 5 shots, unlabeled, classified `other`; also the
  bot's last frame `debug/runs/20261019-171033/frames/00179-screen-other.png`).
- Issues seen in these runs (analysed with `tools.analyze_run`), proposed fixes NOT made yet,
  waiting for the user's OK:
  A. `170812` tap #4: r1c3 cream at **0 % exposed** (opener rule, score 0.55) beat r1c2
     darkgreen (25 %, 0.54) and r0c4 darkred, which opens the same cell r1c4. Cream then stalled
     31 s (lost slot). Fix idea: a cost for taking a slot with low exposure, so an opener at
     ~0 % loses to a move with exposure that opens the same cells. Asked the user: is tapping a
     buried colour ever right just to open a path?
  B. `170918`: every candidate was negative (chest risk). It tapped #2 magenta (-0.11) and, as
     a **burst** (unconfirmed), #4 magenta (-0.19), which opened the second chest. Fix idea:
     burst only moves with score > 0 that don't open a chest. Asked the user: when every move is
     negative and a belt bucket still fills, should the bot wait instead?
  C. `170918` tap #2: a false "no visible effect": the check ran 1.1 s after the tap (bucket
     still in its cell) and k fell 3->2 because pink finished at the same moment. Two in a row
     would stop the bot. Fix idea: `bot._verify` waits ~2 s for a change and counts finished
     belt buckets.
  D. The chest-opening flash is logged and saved as an unknown bucket colour (LAB 98/-18/68)
     every time. Fix idea: don't report unknown colours on a chest cell while it opens.
  E. Add a `level_complete` template from the win shots (WIN detected, not inferred).
- L186 mechanics added (user-confirmed, see GAME_RULES.md): **chest N** (opens when connected,
  scatters N buckets; risk-priced, never forbidden), **mystery bucket** (revealed by tapping a
  4-neighbour; small bonus), **wall** (panel-coloured hole = `x`, blocks paths).
- Earlier the same day: buckets under a just-opened key lock are tappable at once
  (`hints.RecentUnlocks`); a colour stalled on the belt = lost slot (wait while the belt
  collects, else a penalised last resort; jitter-proof stall clock).
- **NEXT**: the user's live check of phase 6; answers on A / B, then fixes A-D.

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

## Reach band + waiting must have a reason (2026-10-16, live L183 shot 056) — done, awaiting OK
- Live: 1/5, only r1c1 cyan tappable (opens a path), bot waited "avail 2.6% < 3%".
- Reach: `sand.contact_layer` = per picture column the lowest `vision.contact_layer` (0.022,
  ~28 px) of SAND above the bottom edge, searched within `contact_reach` (0.07); was a 10 px
  band. 056 cyan 2.6 % -> 15.5 %; 052 darkred still on top (41 %), red 12 % but claimed by
  the red belt bucket (avail -0.7 %, rejected).
- Waiting: `strategy.belt_idle` / `min_exposure` (GAME_RULES "When to wait"): idle belt ->
  never wait; 3+ free slots -> no exposure needed; openers (useful opened bucket / path step,
  2+ free) -> 0; last slot strict. Path planning now runs even when no candidate passes, and
  the first step of the best path is re-evaluated with the opener minimum.
- Saved-shot decisions: only two changed from wait to a tap - 056 (tap r1c1 cyan, path to
  r4c1 darkgreen) and 031 (L181, empty belt: tap r1c2 red tower). Shot 056 added
  (LABELS.md), tests for every rule.

## Live dry-run after all of the above (2026-10-16 ~19:06, L183) — mostly superseded
(The extra lattice row r7 below is the same symptom as the L186 freeze: `hints.LatticeLock` now
ignores extra bottom rows within a level. The unknown sand colour item is still open.)
- Good: ~20 confirmed moves, no pointless waits; path moves live (`tap r3c1 orange, path to
  key red: step 1/2`, `tap r4c1 darkgreen, path to r5c0 darkgreen: step 1/2`); open light
  green read right (`tap r4c0 green`); "hint arrows: 4 masked out" in 6 analyses. The waits
  ("no bucket passes the rules" x3, "conveyor full") were with a nearly full belt.
- **Hint arrows sometimes NOT detected** (2x "wait: no tappable bucket"): no "masked out" log in
  those analyses and the old misread pattern came back (lattice with a row r7, `r7c2
  orange-up`, face-up creams "path rule says closed"). Suspect the arrows animate (bob / pulse /
  fade in) and the template score drops below `vision.hint_threshold` 0.8 (non-arrow max on
  the dataset 0.56). No frame saved (no report taken) -> not tuned blind.
  NEXT (waiting for the user's choice): either the user presses ctrl+alt+s when it happens, or
  the overlay auto-saves the frame when the lattice shape changes between analyses
  (e.g. 7x7 -> 8x8). Then: multi-frame arrow templates / lower threshold / a lattice
  sanity check against the last stable lattice.
- Unknown sand colour LAB 84/-6/-13 (pale blue-white, ~5 % of the picture) saved at the start
  of that run: `dataset/unknown_colours/20261016-190600-107-sand-84_-6_-13.png` (untracked,
  not committed). Logged next to a non-gameplay screen: a new picture or a transition frame?
  Ask the user; add with `python -m tools.add_colour` or delete.
- Many single-frame "looks face-down but the path rule says playable" warnings right after
  moves: flip animation in progress, expected (never tapped; 2-frame confirmation).

## Phase 5 — bot.py plays one level (2026-10-17) — done; live-tested once on L186 (2026-10-19)
- `bot.py`: capture -> screen type -> analyse -> 2-frame `DecisionGate` -> `actions.Tapper`.
  Failsafe corner checked every loop + before every tap; ctrl+alt+p / ctrl+alt+q; rate limit;
  only inside the screen rect; geometry re-resolved right before tapping (frame size must match
  the analysed frame, else no tap). `--dry-run`, `--max-taps N`, `--countdown S`.
- After taps: gate reset, `tempo.ReleaseLedger` (no re-tap for `retap_guard_s`; towers keep
  their later stages on the belt), settle on `bot.settle_zones` (grid + counter) only, then the
  first gameplay frame verifies the tap changed the cell / k (`max_no_effect_taps` -> stop).
- Tempo (`tempo.py`, `config.yaml bot:`): up to `burst_max` taps from one confirmed decision,
  each re-chosen by `strategy.choose` on the simulated state, tappable in both confirming
  frames, >= `burst_min_free` free slots before it (never the last slot), avail >=
  `burst_min_avail`. "About to be covered" = contact-layer share of a colour not on the belt
  dropped >= `cover_min_drop` within `cover_window_s` -> `weight_covering` x drop bonus
  (bot heuristic, not a game rule).
- Non-gameplay `screen_confirm_frames` x in a row -> stop: fail / revive / level_complete = win;
  other screens with < `win_sand_max` sand left = inferred win (no level_complete template).
- Run log `debug/runs/<stamp>/` (`runlog.py`): run.json, events.jsonl, belt.csv, frames/,
  result.json, bot.log. Post-mortem: `python -m tools.analyze_run [run_dir]`.
- `Analyzer.analyze(adjust=...)` hook (the ledger amends the state before `choose`).
- Analysis ~0.85 s/frame (screen classification ~0.7 s), so a confirmed move takes ~1.7 s.

## Dry run 20261017-103804 (user playing, bot watching) + fixes (2026-10-19) — done, awaiting OK
- Of the user's 18 moves the bot had picked the same cell for 11, the same colour for 2.
- **Unlocked buckets**: right after a key opens a lock the bucket under it reads face-down
  (padlock animation) and was refused; the user confirmed they are playable at once.
  `hints.RecentUnlocks`: for `vision.unlock_trust_s` (10 s) after a key-lock cell reads as a
  bucket / empty, the path rule alone decides (`GameView.unlocked`).
- **Stalled colour**: cream sat at 99 % but the stall clock kept restarting (labels 99/98, frames
  without cream), so a second cream (path step r4c5) was chosen. `BeltTracker` now needs
  `stall_min_growth` points to count growth and keeps unseen colours for `stall_forget_s`.
  Strategy: stalled colour = lost slot - rejected while the belt collects, else a penalised
  last resort (`weight_stalled_colour`); planner path cost `path_cost_stalled`.
- Regression tests: `tests/test_run_20261017.py` on 5 frames in `tests/data/run-20261017-103804/`,
  belt jitter replay in `tests/test_state.py`, rules in `tests/test_strategy.py`.
- First live run done (L186, below).

## First live run 20261019-162703 (L186) + chests / mystery / walls (2026-10-19) — done, awaiting OK
- 9 good taps, then frozen at 4/5. Cause: not the path rule. In runs of frames the boosters below
  the grid joined the grid region, the lattice got a 7th row with a smaller pitch and every cell
  was read across two rows; the 7-row frames chose nothing, the correct 6-row frames r4c0
  darkgreen, so the 2-frame gate never confirmed. Fix: `hints.LatticeLock` (the level's grid
  geometry; `Analyzer.reset_level()`, called by bot.py / overlay), regression with a synthetic
  booster bridge on shot 058. Watchdog: STUCK log + frame after `bot.disagree_report_s`.
- New cell types (GAME_RULES "Chest", "Mystery bucket", "Wall"): `chestN` / `mystery` locks in
  `cells.CELL_LOCKS` from palette ui colours `chest_wood` / `mystery_grey` (built from shot 057);
  walls are panel-coloured holes = `x`. Grey-shape templates did not work (the sprites are drawn
  in perspective: left and right chests correlate at 0.2).
- Lookahead / strategy: chests open when connected (risk penalty `planner.scatter_risk`, plans end
  at a chest), mystery reveal bonus; Analyzer waits `chest_settle_s` after a chest opened and
  re-reads. `strategy.py` was split: `analyzer.py` (Analysis, Analyzer, DecisionGate).
- Dataset: shots 057 (L186 start) / 058 (the frozen state) annotated; palette rebuilt (new ui
  colours; a few bucket looks refined by the new samples). Tests: `tests/test_run_20261019.py`,
  `tests/test_chests.py`, frames in `tests/data/run-20261019-162703/`.
- Open: no frame of the actual 7-row misread was saved (frames are saved on decision changes
  only); the regression uses a synthetic bridge. The chest-opening flash reads as an unknown
  yellow (`dataset/unknown_colours/` 20261019-*): not a bucket colour, nothing to add.
