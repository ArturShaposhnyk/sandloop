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

Branch: `claude/compassionate-dijkstra-osnm1j`. Tests: 199 passed, 1 xfailed (screen 020, see below).

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
- **Tower stage lock**: a revealed tower stage looks face-up but stays locked until the stage
  above reaches 100% on the conveyor (031 r1c2 red right after the cream top was tapped). One
  frame can't tell; the bot must remember tapped towers across frames (phase 5).
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
  lookahead (cell empties, a tower keeps its next stage locked, ice counters count down).
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
  - NEXT (before phase 5): run `python -m tools.overlay --dry-run` while actually playing
    (belt moving, level end / retry, ads), then inspect `debug/vision_fail/fails.jsonl`
    (step, failure, frame diff) and the saved frames, find the cause, fix it.
- Env note: `.venv` is Python 3.9.6 (Xcode); CLAUDE.md says 3.11+. Works (future annotations).

## Phase 5 — next
- `bot.py`: full loop on one level (stable frame -> Analyzer -> tap via actions.Tapper with all
  safety checks), log every move (frame + decision) to `debug/`.
