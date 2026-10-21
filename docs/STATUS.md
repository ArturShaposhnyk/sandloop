# Sand Loop bot — status / handoff

Read this first, then `docs/GAME_RULES.md` (game mechanics, source of truth) and `PROMPT.md`
(full brief and phase plan). Work phase by phase; after each phase stop and wait for the user's
OK. Branch: `claude/compassionate-dijkstra-osnm1j` (pushed).

## Done (phases 0-6, phase 7 started)
| Phase | What | Commits |
|---|---|---|
| 0 | setup, config, permission check | `3b5b8f7` |
| 1 | capture (window + screen rect, hover chrome handled), taps, overlay | `3b5b8f7`, `ba16f6d`, `9cf1d40` |
| 2 | `tools/collect.py`, labeled screenshots in `dataset/screens/` (`LABELS.md`) | `385c523`, `015ab26` |
| 3 | `screens.py`, `vision.py`, `cells.py`, `palette.py`, `digits.py`, `tools/build_palette.py` | `5d8dd41` |
| 4 | `state.py`, `strategy.py` (greedy + path planning), `tools/overlay.py --dry-run` | `6ff1c26`, `9f2b4e7`, `774d07e`, ... |
| 5 | `bot.py` plays a level with real taps | `4b1883a`, `4f297f2`, `00daa0e` |
| 6 | screens outside gameplay (`flow.py`, `levels.py`, `adclose.py`, `hud.py`), `--levels N`, session limits | `a07c297`, `c1b1423` |
| 7 (started) | lookahead 2-3 taps (`lookahead.py`), sand above the contact layer (`sand.contact_layers`), every level -> `debug/levels.jsonl` | `ebd0c2a` |
| 7 | whole-level plan (`sim.py`, `levelplan.py`), level memory, per-colour capacity, endgame reserve, sacrifice | `ee2c1c8` |
| 7 | run 161536: measured belt speed in the simulator, plan waits never stall, plan checks, focus / window fixes | `b5d5a7e` |
| 7 | run 171133: sand read to the picture's bottom edge, key-colour reserve, plans ranked by dead buckets, ad skip | `6198f3b` |
| 7 | run 224631: ✕ on an ad disc, frontmost check; gamble / plan / tower sand rules | `8f29be2`, `819daac` |
| 7 | L188 shells + picture locks, L189 chain lock, lattice fit by wall runs | `96363d3` |
| 7 | first L187 win (run 000105); finished board no longer starts a level | `66a279d` |
| 7 | belt speed 2x (▶▶2x tapped at level start, planner 2x fill speeds, 1x / 2x calibration), ad chains, key colour on the last slot | `fdc832d` |
| 7 | L189 run 092045: ice needs its number, ice slot reserve, lost-position finish, booster arrow row | `bdc4c98` |
| 7 | L189 run 101655: popup screen, level memory (record order, face-free signature, whole-level bans, colours under locks, clean end reads), player hints in the planner, 2x belt speeds | `44fdfc0` |
| 7 | run 112241: L189 WON; no phantom level / inferred win; L190 tunnel model | `f59c683` |
| fixes | test clock (`85e5960`); L187 lattice / colours / locks (`76ea470`, `5750651`); unlock rule + this handoff (`c9e26e2`); lattice never replaced mid-level + open-tower foot (`428a308`); last-slot gamble + flying bucket (`ebd0c2a`) | |

Tests: `python -m pytest -q` - all 840 pass.
The shared test Analyzers use a frozen fake clock, so results never depend on machine load.

## Where it stands (2026-10-21)
### Run `20261021-112241`: **L189 WON** (attempt 2), then a false win
- The start plan used the memory ("4 failed sequence(s) avoided, capacity from earlier
  attempts, r3c1 cyan/cream seen under its lock before"); attempt 2 won at 421 s.
- **False "LEVEL 3: WIN (inferred ...)" at 430 s** - it was not L190: after NEXT and the "take a
  break" ad, the finished L189 board faded back in (frame `tests/data/run-20261021-112241/
  00470-finished-board-faded.png`): no sand, the empty cells dimmed (LAB 80/-1/-13), read as
  "two objects" / unknown colours (also the r4c6 / r5c6 "two objects" the user saw), so the
  finished-board check (all cells empty) failed, a level started and the next screen change
  "won" it. Fixed: (1) a board without sand (< `bot.win_sand_max`) never starts a level; (2) an
  inferred win needs this level's own taps and a start with >= `bot.infer_win_start_sand` (5 %)
  sand. The run therefore never saw L190 - no L190 frame exists yet.
- **L190 tunnel / dispenser** (user-confirmed, docs/GAME_RULES.md "Tunnel"): lock cell
  `tunnel` with number + opening side (`Cell.opening`, `Lock.opening`, `Lock.front()`); a wall
  for the path rule; taking the bucket in front of it drops the next one (UNKNOWN, worst case:
  the cell stays blocked, the plan's horizon) and counts N down; N counts as buckets still to
  come. Recognition is template-driven (annotation `tunnelN:<side>` -> `tools.build_palette`
  -> `dataset/templates/icons/tunnel_<side>-*.png`, matched like lock icons, no colour needed),
  but **no template exists yet**: on L190 a tunnel reads UNKNOWN (never tapped, an obstacle,
  saved to dataset/unknown/). Needs L190 screenshots (see NEXT).
- `state.after_tap` now counts the "taps" locks down from the state after the keys / chests
  that were already flying (was: the state before them; same results in every test).
- Tests: `tests/test_run_20261021_112241.py`.

### L189 run `20261021-101655` (3 attempts, all lost; flow between them perfect)
- **The three attempts (levels.jsonl)**: each tapped the same 15 buckets - the whole top group,
  the only reachable ones; L189's ice is 15, so it opens only with the 15th tap - in almost the
  same order (r0c0 cyan, r0c1 cream, r0c6 cyan first every time; r0c3 red, r2c1 darkred, r2c2
  red last). Each ended the same way: 4 buckets of buried colours (red / yellow / darkred,
  under cyan + orange) on the belt, the 15th tap (a lost-position tap, belt stuck 16-20 s)
  broke the ice and filled the belt, the cyan under it (attempt 1's end read: r3c1 cyan-up) had
  no slot. Attempt 3 at the 15th tap: belt yellow 73, darkred 92, red 92, cream 70.
- **Why the level memory did not make them differ** (all fixed):
  1. the plans were never stored: `levels._level_left` reset the analyzer (and its PlanRunner)
     before `_record_level` ("plans": [] in all three records);
  2. attempt 3 had its own board signature: its first frame (after an ad) read the top row
     face-down -> no memory at all. The signature now ignores faces and is recomputed from
     each record's start grid (old records match again);
  3. bans compared a plan's OWN taps (counted from the re-plan's board) with whole-level
     sequences, and plans were 9-11 taps long: nothing was ever banned after the first re-plan,
     and the greedy strategy (most taps after the plan ran out) never looks at the memory.
     Plans are now prefixed with the level's taps (`PlanRunner.done`, fed by bot.py);
  4. attempts 2 / 3 ended on a frame read under the revive popup (board "green", sand ~0):
     that "end" went into the capacity correction. End reads with colours the start never
     shows are skipped (`levelmemory._clean_end`), and the popup is no gameplay frame any more.
- **Popup** (user: skip board analysis while a popup is shown): the revive popup sliding in
  (frame 00507, "Keep Playing" 0.49) read as gameplay. `screens.popup_bar`: a bright green bar
  >= 0.65 of the width over >= 1.5 % of the height -> screen "popup" (no board read, no tap,
  not a level end by itself). 0 false positives on every saved tap frame and test frame.
- **Player hints -> planner** (docs/GAME_RULES.md "Level plan", `config.yaml planner:`):
  1. timing: `weight_idle` 0.002 per belt slot-second of a bucket filling < 2 %/s (the sim
     predicts when a colour reaches the contact layer) - plans no longer wait 600 s on a
     creeping darkred, and release buried colours later;
  2. further ahead: `weight_tap` 0.03 per bucket released, start plan beam 24 / 6 s (was 8 / 3 s);
  3. go deep: `weight_open` 0.05 per cell opened (ice, key lock, chain gate, mystery); colours
     seen under locks in earlier attempts (`levelmemory.revealed`, new `revealed` field in
     levels.jsonl, `Analyzer._watch_hidden`) open to known buckets in the simulator, so on Try
     Again the plan reaches past the ice (L189: r3c1 cyan, from attempt 1);
  4. play fast: back-to-back plan steps (up to `burst_max` 3) are tapped in one burst
     (`Candidate.follow`, each tappable on the frame and passing the sand rules / reserves),
     a follow-up costs `burst_tap_s` 1 s in the simulator. L189 start plan now: r0c0, r0c1,
     r0c5 in one burst.
- **Belt speed 2x calibrated** (645 samples, 4 runs): `speed_x2_fill_*` 90 x (share - 0.03),
  max 60 %/s (was the doubled 1x guess 200 x (share - 0.08): nothing below 8 %).
- **Honest limit - the simulator is too pessimistic for L189**: from the start frame, even
  wide searches (beam 256, any ranking I tried, measured capacities) end at the 15th tap with
  the belt full (red / yellow / darkred at 0-45 % under cyan + orange), while the real belt at
  that point was 70-92 % full. Its picture model (vertical columns, a 4-cell contact layer)
  does not reproduce how the real sand pile slides; the plan therefore stops at 13 of 15 taps
  and the live ice reserve / strategy decide the rest. Next step if L189 keeps failing: fit the
  contact depth / sliding against the run's belt.csv + frames (plan checks in levels.jsonl now
  hold the predicted vs real belt).
- Tests: `tests/test_run_20261021_101655.py` (frames 00005, 00507, the 3 attempts trimmed);
  `tests/test_speed.py` checks the measured 2x values.

### L189 run `20261021-092045` (14 taps, stopped by the user after 200 s on a lost board)
- **User's analysis**: belt 4/5 (red 23, yellow 27, darkred 8, red 57), the picture's bottom is
  cyan / cream; the cyan bucket needed is under the ice "1" at r3c1, and the tap that breaks
  the ice fills the belt: no winning move. The mistake was earlier: keep 2 free slots (1 for
  the tap that breaks the ice, 1 for the bucket under it).
- **Ice read** (fix 1): the ice "1" read as a face-down cream/cyan tower (thin digit: 0.040
  dark pixels < `ice_min_dark` 0.05) -> "STUCK ... misread grid?" (frames 00113-00257); the
  cyan bucket r4c6 read as ice without a number. Ice now needs its number (`CellLock`
  `counter="required"`; `classify_cell(skip=...)` re-classifies without it, replacing the chest
  `wood_frac = 0` hack), `ice_min_dark` 0.03. Run frames: ice5 / 3 / 1, r4c6 cyan-down. The ice
  in L189 shows no colour of the bucket under it (plain ice); the cell stays "ice N", colour
  unknown (the plan ends there: horizon). Frame 00103 (counter morphing 2 -> 1) still reads
  r3c1 "two objects" for a frame.
  Note: the planner's plan at 49 s had "tap r3c1 cream" - the same ice misread.
- **Ice slot reserve** (`endgame.ice_reserve`, in `strategy.evaluate` and the plan runner's
  `_reserve`): ice N reachable with b buckets in front -> keep max(N, b) + 1 slots; a tap that
  leaves fewer only for a bucket that fills and leaves (exposed >= `strategy.ice_live_exposed`
  0.08, can fill, not stalled); only while max(N, b) + 1 <= slots. Replay: #12 red (3 %, ice 4,
  3 free) and #13 darkred (6 %, ice 3, 3 free) are now rejected ("ice r3c1 (4): keep 5
  slot(s) ..."), the bot waits for the belt instead.
- **Lost position** (`endgame.lost_finish`, `strategy.lost_stuck_s` 15 s): belt stuck that
  long, nothing passes, no last-slot gamble -> "lost position: tap rXcY ... - then Try Again"
  (tap event `lost: true`; never a burst follow-up). The loss is recorded by the existing
  revive / fail flow (levels.jsonl, level memory). Final board: taps r2c2 red after 15 s.
- **Booster arrows** (fix 3, "unknown cell r5c2 (two objects)"): the 4 hint arrows over the
  bottom row matched 0.60-0.79 (frame 092236: none >= 0.8), so r5c2 (chain padlock) read two
  objects and r5c6 (yellow key) read EMPTY. `hints._in_row`: a match >= `vision.hint_row_threshold`
  0.65 counts when another of the same scale sits at the same height -> all 4 masked, row 5
  reads right. Also fixes L187 frames 00143 / unknown-125531-186 (run 125351): they now read
  the level's 7x7 by themselves (tests updated).
- Tests: `tests/test_run_20261021_092045.py` (frames 00078, 00083, 00094, 00238, two arrow
  frames from dataset/unknown); `tests/test_lookahead.py` gamble tests run with the lost finish
  off (`NO_LOST`) - their belts are stuck 30-50 s, which now ends in a lost-position tap.
- Not done / open: the planner (sim) does not model the ice reserve itself - its steps are
  checked by the reserve at tap time (skipped + re-planned). The bucket under the ice has no
  colour until it opens.

### Handoff summary (before /clear, `721de47` + this note)
- **L187 (HARD) won** live (run `20261021-000105`, 29 taps, plan followed to the end); the
  finished board after a win no longer counts as a new level (`66a279d`).
- **Belt speed 2x**: the free ▶▶2x button (resets every level) is tapped at level start
  (`bot.speed_x2`), the planner uses 2x fill speeds (`tap_s` not scaled, user agreed); 2x
  calibration still to do (`python -m tools.belt_speed <2x run>` -> `planner.speed_x2_fill_*`).
- **Ads**: chains of skip ▶| / ✕ (incl. a small ✕ on a yellow disc) tapped every 3 s until a
  known screen (`flow.ad_retap_s`, cap `ad_chain_max_s` 10 min); never INSTALL / Download / Get.
- **Endgame rules (user)**: gamble only buckets that can fill (or open one that can), plan
  steps the sand rules reject are skipped + re-planned, towers only when the top can fill, key
  colour may take the reserved last slot at 20 %; margins stay 1.0 (run 171133 evidence).
- **L188**: shell buckets (tap only with a shell of their colour at the bottom), shells in the
  picture, picture locks (keys under the curtain, confirmed; box sand masked); **L189**: chain
  lock gate + magenta lock / key. Details below; mechanics in docs/GAME_RULES.md. Not yet
  live-tested: L188 / L189 code. Open questions for the user: listed at the end of
  GAME_RULES.md (one shell per shell bucket? shell at the bottom with no bucket waiting? the
  chain lock as one 2-cell gate?).
- NEXT: live `python -m bot --levels 3` from L188 (see NEXT below).

- **L188 / L189 mechanics (user-confirmed, docs/GAME_RULES.md)**. L188 run `20261021-001541`
  stopped by the user: the grid failed "grid geometry 6x4: walls off its cell boundaries (fit
  0.77)" - the board WAS read right (4 columns + 2-cell top lane); blue / pink (shell) bucket
  pixels near the panel colour scattered short edges. `vision.wall_alignment` now counts each
  edge by its longest unbroken vertical run: every L188 frame >= 0.88 (most 1.0), misread L187
  4x4 still <= 0.51. 2x worked in that run too.
  - **Shell buckets**: `icons.ShellMatcher` (shell outline on the gradient image, templates
    `dataset/templates/shells/` from annotated `+shell` cells of 065 / 066), threshold 0.84
    (L188 shells 0.86-0.99 live; L189 plain cyan dome 0.81); `cells._shell_bucket`: grey shell =
    face-up, shell in the dome's colour = face-down, colour from the saturated pixels; checked
    before mystery (grey shell = mystery grey). Strategy: a shell bucket is rejected ("shell
    bucket: no pink shell at the bottom of the picture yet (lowest 6.2% up)") until a shell of
    its colour is within `strategy.shell_near_bottom` (3 %) of the bottom; plan steps too.
    Known gap: 067 r5c3 (darkred shell next to the opening curtain's gold frame) scores 0.79 and
    reads cream/darkred.
  - **Picture shells / locks** (`picture.py`): shells by masked outline + colour purity + the
    colour of a shell bucket in play (no search without shell buckets); depth to the picture's
    bottom edge. 065: all 6 found; 066-068 miss the blue shell touching the darkred one (the bot
    then waits longer); other levels with L188's shell colours <= 0.79 (threshold 0.76).
    Locks: keyhole template (0.96-1.0) + the box's dark grey frame; colour from the keyhole
    ring (yellow / red). The boxes are masked out of the sand reading (065 contact layer now
    orange 55 % / darkred 34 % / cream 11 %, was cream 64 %; the "unknown colour LAB 73/10/-2"
    was the overlay). A key whose colour has a picture lock opens it (`pics_opened`,
    `weight_pic_unlock`). Analyzer logs once per level "picture lock yellow: its key is not on
    the board yet (under a curtain / lock / ice?)" - or "unknown mechanic, ask the user" if
    nothing hides one.
  - **Simulator**: shells are markers in the column stacks that fall with the sand; a shell
    bucket waits (slot taken, no fill) until its shell is down; shell buckets are moves only
    with their shell down; lock boxes are LOCKED cells (never collected); an opened picture lock
    ends the plan (horizon). L188 start plan: r0c1 darkred, r1c1 orange, r2c1 cream, r3c1 blue
    (the column the user cleared first), then horizon (curtain).
  - **Chain lock (L189)**: cell lock `chain` (icon templates from shot 072), a gate: impassable
    until its key flies, then EMPTY (`state._release_keys`, `gates_opened`); adjacent chain
    cells are one gate with the padlock's colour; `LevelState.behind_gate`, planner key target
    worth the buckets behind it. Magenta lock / key colour added (palette keys: one look per
    shot). Shot 072 is a Mac screenshot: used for shapes / key colours only, not for bucket
    colours (`no_bucket_colours`: its yellow was b 88 vs 72 and turned L187's fading yellow
    padlock into orange).
  - Dataset: the user's L188 recording renamed 001-007 -> **065-071** (seqs clashed); 065 / 066
    annotated (grids with `+shell`, `picture` shells / keyholes / boxes); 072 = L189 start.
  - **Mismatches / open (asked)**: no frame shows a shell dropping in; one shell per shell
    bucket? a shell at the bottom with no shell bucket waiting? the chain lock spans two cells
    on screen (modelled as one 2-cell gate).
- **FIRST L187 (HARD) WIN: run `20261021-000105`** (`d566114`, `--levels 3`), after 6 lost
  attempts: 29 taps (31 with the speed tap), level plan followed to the end (20 plans, 2 steps
  skipped by the new sand rule: r1c6 cream avail -11.7 %, r4c0 orange last slot 10.9 %; no
  gamble), won at 163 s, then NEXT and the ad ✕ worked. **Belt speed 2x worked**: read off at
  10.9 s, SPEED TAP at 11.3 s, "belt speed 2x: on" at 13.4 s, re-planned at 2x; every plan from
  14.7 s on says "belt 2x".
- **Bug fixed - one win counted 3x**: after the win / ads the game showed the finished L187 board
  (empty grid, 0/5, no sand: frames 00272, 00305); each frame started a "level" that ended at
  once as an inferred win, so "--levels 3" stopped on the menu (L188 / PLAY) after one level.
  Now a level starts only on a board with something to play (a bucket / lock / unknown cell, or
  sand >= `bot.win_sand_max`); such a frame logs "gameplay frame with an empty board and no sand
  (the finished level) - not a new level" (event `finished_board`) and resets the Analyzer
  (no lattice lock / empty capacities carried into L188). Test:
  `tests/test_run_20261021_000105.py`. Note: `debug/levels.jsonl` holds the two phantom L2 / L3
  "win" records of this run (empty board signature; harmless for the level memory).
- **Belt speed 2x** (user: the ▶▶2x button is free and resets every level): `hud.read_speed`
  (green share of `bot.speed_zone` -> on, grey -> off; live gameplay frames: on 0.34-0.57 green,
  off ~0.4 grey; dataset shots 006-056 / run 103804 were played at 2x, every bot run so far at
  1x), `hud.SpeedTracker` (2 equal readings). Log "belt speed 2x: on / off"; level record
  `speed_x2`; belt.csv column `speed`. `bot.speed_x2` (true): while it reads off at level start
  the bot taps the button centre (only it; "SPEED TAP ▶▶2x", frame `*-speed-tap`), at most
  `speed_max_taps` (2) per level `speed_retap_s` (3 s) apart, no bucket move meanwhile; still
  off -> "belt speed 2x still off after 2 tap(s) - playing at 1x". Planner: a speed change
  re-plans ("level plan: belt speed 2x - re-planning"); `sim_config(x2)` scales fill_rate /
  fill_max by `planner.speed_x2_factor` (2) until `speed_x2_fill_*` are calibrated. **tap_s is
  not scaled**: the simulator's times are real seconds and tap_s is the bot's own cycle
  (~4.2 s); scaling both would count 2x twice. `python -m tools.belt_speed` fits 1x and 2x
  separately (older runs = 1x).
- **Ad chains** (user): ✕ / skip taps are unlimited (not counted towards `same_button_max` /
  `max_taps_between_levels`), repeated every `flow.ad_retap_s` (3 s) until a known screen;
  `flow.ad_chain_max_s` (600 s) without one stops the bot. Taps stay on corner ✕ / skip glyphs
  only (never INSTALL / Download / Get).
- **Key-colour reserve, run 224631 (user asked why no slot was kept for yellow)**: before #26
  (2 free) yellow was no key - 18.5 % of the contact layer < `key_min_exposed` 20 % and it
  covered only 0.2 % of the buried blue; as a key it needs 1 slot (nothing in front), which
  cream left anyway. At #27 (1 free) yellow was a key (22.9 %) and the reserve did keep the last
  slot, but the last-slot minimum (25 %) rejected yellow itself. Fixed: a key colour's own
  bucket takes the last slot at `key_min_exposed` (20 %). At #27 the lookahead then still
  calls the belt jammed (yellow 3.82 % left < a 4.07 % bucket at `look_sand_margin` 1.0).
  **Margins stay 1.0** (user, 2026-10-21, after the evidence): 0.9 was proposed because sand
  reads ran low, but run 171133 #54 tapped yellow with 3.88 % left vs a 4.06 % bucket (95.5 %)
  and that bucket ended at exactly 95 % - the read was right; the darkred / cream under-reads
  were the cut picture edge (fixed in `6198f3b`). At 0.9 the bot repeats that tap (171133
  regression test). `sufficiency_margin` (0.5) untouched. tap_s not scaled at 2x (user
  agreed). Lookahead-jam moves now count for the last-slot gamble too.
### Earlier handoff summary (2026-10-20, `d1cd250`)
- **Planner** (`levelplan.py`, `sim.py`, `planner:` in config.yaml): beam search over tap / wait
  sequences at level start and on every unmatched board; ranked win > fewer dead belt buckets >
  sand collected; full belts of dead buckets dropped; unread belt buckets keep their slots; a
  step that breaks the endgame / key reserve is not tapped; waits give up after 5 s without
  sand gain or 5 s behind plan -> re-plan, else the greedy strategy. Plan checks (predicted vs
  real belt) go to bot.log and `levels.jsonl`. Hidden colours end the plan ("horizon").
- **Level memory** (`debug/levels.jsonl`): Try Again = same board signature; failed sequences
  are banned, per-colour capacity corrected from what the attempts used.
- **Belt speed model**: fill %/s = `fill_rate` 100 x (contact share - `fill_min_exposed` 0.08),
  capped at `fill_max` 30; `tap_s` 4.2. Fitted on runs 142035 / 145310 / 161536 with sand cut at
  the zone bottom - **re-fit after the next run** (sand is now read to the picture edge).
- **Ads**: known anchors -> ✕; anchorless screen within 120 s of a screen tap -> ✕ or the ▶|
  skip disc top right (user-confirmed for Trade Republic); never Download; wait 3 min, frame
  every 10 s.
- **Open on L187**: (1) not yet checked live whether the darkred fix + key reserve win the
  endgame; (2) the Netflix ad showed no ✕ / skip - how it ends is unknown; (3) the user's answer
  on whether the first ad (window lost 1.2 s, then splash) closed by itself is still open, and
  whether some ads need waiting for a ✕ / close by themselves; (4) belt fit uses old sand reads;
  (5) a `darkgreen/cream` belt read may be wrong.
- Untracked `dataset/unknown/`, `dataset/unknown_colours/`: unlabeled crops from runs, kept out
  of git on purpose (never committed this session).

- **L187 run `20261020-224631`** (`--levels 3`, `d1cd250`): lost once more in the endgame (27
  taps, 6th attempt), then the revive -> "give up" -> Try Again flow worked, the Kingshot ad
  was skipped (▶|) and its App Store sheet closed (✕), but its end card (small white ✕ on a
  yellow disc top right, 0.017 x the screen width) was never found: 100 s "unknown screen"
  until the user quit. **Fixed**: `adclose.find_skip` also accepts a ✕ on such a disc
  (`CloseMark.kind`, logged "ad ✕ on a disc"). **Fixed**: "iPhone Mirroring is not frontmost
  2.0s after activating it" before every tap from 138 s on (+2.6 s per tap) while every tap
  worked: `NSWorkspace.frontmostApplication()` is stale without a run loop (verified: it kept
  saying "Code" after Finder was activated); `actions.frontmost_pid` now pumps the run loop.
  **Endgame (user rules, done)**: #27 the last-slot gamble took purple r5c6 (3.2 % sand
  left < a 3.67 % bucket) because it uncovered the most for the belt; #24 the plan tapped the
  orange/darkgreen tower the rules rejected (avail -5 %, orange 6.6 % left < 7.1 % needed).
  Now: (1) the gamble only takes a bucket that can fill (`endgame.can_fill`: sand left >=
  `gamble_fill_margin` 1.0 x (belt need + 1 bucket)) or opens the path to one; fillers first,
  "uncovers" breaks ties. (2) a plan step the sand rules reject (exposure as a path step, sand,
  tower top) is skipped and re-planned with it forbidden on that board (`PlanRunner._skip`,
  log "skipped: ... - re-planning without it"). (3) a tower only when its top can fill
  (`evaluate`: "tower: X cannot fill"; never a sacrifice). Replay: #24 skipped; #27 gambles
  nothing - at that moment yellow had 3.82 % left vs a 4.07 % bucket (my earlier "~1.6
  buckets" was the 141 s figure); with `gamble_fill_margin` 0.9 it gambles yellow. Run
  142035's end now gambles cream r1c6 (opens purple r3c6) instead of darkred (1.2 % left).
- **L187 run `20261020-171133`** (`--levels 3`, `d130e6d`): lost twice in the endgame, then an
  ad stopped the session. Endgame: belt full of buckets at 96-99 % while darkred r3c2 (tappable:
  r3c3 is empty, so not 2 slots away) was rejected "not enough sand (0.0 %)". **Root cause of the
  darkred / cream under-read since run 145310**: the picture reaches ~12 px below the
  `vision.picture` zone and those colours settle at the bottom. `sand.read_sand` now reads to
  the real bottom edge (L187 end: darkred 0.17 % -> 1.34 %); colours down to 0.05 % are
  reported (`vision.sand_min_share`; analyze_run shows "sand left 0.31 % (buried: not in the
  contact layer)"). Replayed: at #54 the strategy now taps darkred.
- **Key colours** (`endgame.key_reserve`, user): keep the slots needed to reach the nearest
  bucket of a colour covering buried belt sand (1 + buckets in front on the path) - only while
  that many are free. **Planner**: ranks nodes win > fewer dead belt buckets > score, drops nodes
  whose full belt is all dead, keeps a slot for each belt bucket the reader did not see
  (k 4 with 3 read), and a plan step that breaks a slot reserve is not tapped.
- **Ads**: an unknown screen within `flow.ad_after_tap_s` (120 s) of a screen tap is searched for
  the ✕ and the ▶| skip disc (`adclose.find_skip`, top right; the mute disc top left never
  matches); never Download. Unknown / ad screens wait 3 min, a frame every 10 s. The second ad
  (Netflix, then black for 60 s) showed no ✕ / skip in its only frame.
- analyze_run "-1 other allowed" (plan steps the rules reject) fixed in bot.py.
- **L187 run `20261020-161536`** (planner on, `d550e5a`): the plan worked for 6 taps and
  re-planned twice, then step 3 "tap r4c4 purple" waited for the belt (k 3 > 2) 32 s: yellow
  37 % and orange 0 % could not fill. Causes: (1) the simulator filled buckets from any sand in
  the contact layer, but real buckets get ~nothing below ~10 % of it (orange was at 7 %);
  (2) the simulator's picture was cut at the zone bottom, not the real picture edge (orange
  26 % instead of vision's 7 %); (3) a plan wait had no limit; (4) the orange stall clock
  restarted every other frame (one or both orange buckets read). Then iPhone Mirroring vanished
  at 76 s (no frame of it saved) and the bot stopped after 10 s.
- **Fixes**: belt speed measured (`python -m tools.belt_speed`: ~100 %/s x (contact share -
  8 %), max 30 %/s, ~4.2 s per tap) and used by `sim.py` (`planner.fill_rate`,
  `fill_min_exposed`, `fill_max`, `tap_s`, `lap_s`, `max_wait_s` replace `lap_take` /
  `laps_per_tap`); the simulated picture is cut at the real bottom edge, block = most frequent
  colour (contact shares now within 5 % of vision); a planned tap into the last free slot costs
  `planner.weight_full`; **a plan wait is given up** after `wait_stall_s` (5 s) without sand gain
  or `wait_grace_s` (5 s) behind the plan's predicted time -> re-plan from the real belt, and if
  that plan waits too the greedy strategy decides until the board changes. Every planned tap
  logs `plan check step i/n: t ..s (plan ..s), belt k .. (plan ..): plan ... | real ...` (also
  in `levels.jsonl` "plans" -> "checks"; `python -m tools.belt_speed` lists them).
  `BeltTracker` no longer restarts a stall clock when fewer buckets of the colour are read.
  From the stalled board (frame 00060) the planner now taps darkred at once (no wait).
- **Focus / window**: live runs activate iPhone Mirroring before the first tap and wait until it
  is frontmost (`clicks.focus_wait_s`, `focus_settle_s`); a tap without visible effect is
  checked on one more frame before it counts. Window lost: the bot waits up to
  `bot.window_lost_max_s` 30 s, saves the last frame (`frames/*-before-window-lost.png`) and
  logs the window list (`window list: ... off-screen ...; frontmost app: ...`). TAP #1 in the
  run did work (the orange bucket flew; frame 00008), the effect check ran on a frame where it
  still showed.
- **L187 run `20261020-145310`** (uncommitted gamble/lookahead code, `bfaad21` + work tree): the
  cream last-slot gamble (#22) opened purple and the sky drained; LOST with the belt 5/5 (green,
  orange, yellow, purple, blue almost done, stuck) while 2 darkred buckets (39 % of the contact
  layer) had no slot. Root cause: darkred sand is under-read (1.0 % of the picture), so "not
  enough sand" rejected darkred from #19 on; #25 yellow was a burst at 0.15.
- **Fixes**: capacity **per colour** measured at level start (`analyzer.colour_capacity`,
  `LevelState.capacity_for`); **endgame reserve** (`strategy.endgame_reserve`); burst follow-ups
  need `bot.burst_min_score` (0.3), no negative lookahead, never a gamble / sacrifice;
  **sacrifice** (user (a)): a sand-short bucket may go when a 2-3 tap continuation revives a
  dead belt colour and a slot stays free. Replay of the run: #20 now taps cream r1c6 (opens
  purple) instead of purple r5c0 0.22; #26 taps darkred, purple waits ("endgame: keep a slot
  for darkred"); no yellow follow-up after #24.
- **Whole-level planner (phase 7)**: `sim.py` (picture as per-column sand stacks, belt laps,
  towers, keys / locks, ice, curtains, capacity per colour) + `levelplan.py` (beam search, live
  `PlanRunner`, `LevelMemory`, `python -m levelplan <frame>`). On by default (`planner.enabled`);
  the greedy strategy takes over at the plan's horizon (unknown buckets) until it re-plans.
  From run 145310's start (with its record as memory: 1 sequence banned, capacities calibrated):
  18 taps to the horizon (the keys fly, the locks show unseen colours), belt never above 4/5,
  0 dead buckets at the end, ~1 s.
- **L187 run `20261020-142035`**: the grid lock held (no lattice change in 155 s), 23 taps, then
  "no bucket passes the rules" for 50 s until quit. Strategy, not vision (analysis in History).
- **Last-slot gamble (user rule)**: belt stuck (every unfinished belt bucket without sand gain
  for `strategy.gamble_stuck_s` 5 s), one slot left, every move rejected by the last-slot
  exposure minimum -> take the slot with the move that uncovers the most sand for the belt's
  colours (`sand.contact_layers` "above" matrix), ties by its own exposure; reason / log
  "last-slot gamble: ...". My reading (please confirm): sand-short moves may be gambled too (the
  run's end had only 0.8 % cream / darkred left), a colour stalled on the belt never.
- **Phase 7 started - lookahead** (`lookahead.py`): per allowed tap the best continuation of up
  to 3 taps (3 best-exposed follow-ups per step) is simulated with `after_tap` and a one-step
  sand model (collecting colours drop the sand above them). Value = (free + live belt buckets)
  / slots; live = enough sand (margin 1.0) and exposed / not stalled, or uncovered by the step.
  Score += `weight_lookahead` x (value - now); a tap that jams a collecting belt is rejected.
  Shown in candidates / overlay text as "lookahead r0c0 purple -> ...: 2 live / 0 dead, 3 free".
- **Level records**: every level (win / fail / unfinished) -> `debug/levels.jsonl`: run, commit,
  strategy config, result, duration, every tap with the decision it came from (state, all
  candidates with scores / lookahead / gamble) and the last gameplay read.
- **Flying bucket**: a cell read empty within `vision.flight_s` (2 s) stays empty when an object
  shows there (`hints.InFlight`, paused while a chest's buckets land).
- **Live wins: L186** by itself (run `20261019-171033`, 21 taps, 118 s). **L187 (HARD)**: two
  live runs got stuck on lattice misreads (`20261020-122638` at 3/5, `20261020-125351` after 21
  good taps: 7x7 <-> 4x4 flip-flop). Fixed offline (below), not live-tested again yet. No L187
  win yet.
- **Lattice lock (2026-10-20, run `125351`)**: the level's grid is now NEVER replaced mid-level.
  The first lattice whose inner walls sit on its cell boundaries (`Lattice.fit` >=
  `vision.lattice_min_fit` 0.8) is locked; for `lattice_lock_frames` (3) frames a finer reading
  of the same grid (more columns over the same outline / fewer rows under the same top) may
  still replace it; after that every gameplay frame is read with it ("frame reads 4x4 ..., read
  with the level's 7x7" logged once per streak). Only a non-gameplay screen (win / fail /
  revive / menu / ad) or another frame size unlocks it. A misfit reading before the lock is a
  failed read (so the bot never locks 4x4 on a frame with hint arrows / sparkles).
- **Open tower in a tall lane fixed**: the lower stage drawn under the bucket crop is read from
  the strip below it (`vision.foot_strip`, `cells.foot_stage`, `vision.foot_min_obj`); ignored
  when the cell below is ice / curtain / chest / mystery / unknown or the same colour (they
  overhang). Checked on every saved live frame: 6 distinct towers found, all real.
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
-5. L190 screenshots: on L190 run `python -m tools.collect --manual --label l190` and snap
   (ctrl+alt+s) the start board + a frame right after a tunnel dropped a bucket; then
   annotate the grid (`tunnelN:<side>` per tunnel) in dataset/annotations.json, `python -m
   tools.build_palette`, and check `python -m vision <shot>` reads every tunnel.
-4. Live L189 again `python -m bot --levels 3` (Try Again uses the memory): check the start log
   "level plan (start, 3 failed sequence(s) avoided, capacity from earlier attempts, r3c1 cyan
   seen under its lock before, belt 2x)", bursts ("burst 2: level plan step 2/..."), "r3c1:
   cyan under its lock (level memory)" when the ice opens, no "green" board at the end, and
   levels.jsonl records with "plans" and "revealed".
-3. Live L189 `python -m bot --levels 3` (from L189): watch "ice r3c1 (N): keep ... slot(s)",
   that ice 1 reads `ice1` (no "STUCK ... misread grid?"), "hint arrows: 4 masked out" with no
   r5c2 / r5c6 unknowns, and on a lost board "lost position: tap ..." after ~15 s followed by
   Quit -> Try Again (second attempt should avoid the banned sequence).
-2. Live L188 (`--levels 3` from L188): watch "shell bucket: no X shell at the bottom", the
   shells / "picture lock" lines (python -m vision on saved frames), "picture lock ...: its key
   is not on the board yet", and whether a shell bucket on the belt fills once its shell drops.
   Then L189 (chain lock + magenta lock).
-1. `python -m bot --levels 3` from the menu (L188 / PLAY): check one level is counted per real
   board, PLAY is pressed after each win, 2x at every level start; then
   `python -m tools.belt_speed debug/runs/20261021-000105 <new run>` -> `planner.speed_x2_fill_*`.
0. Live L187 `--levels 3` with the run-224631 fixes (ad end card ✕, frontmost check, gamble /
   plan / tower sand rules) and 2x: check "belt speed 2x: on" after the SPEED TAP on every
   level start and Try Again; then `python -m tools.belt_speed debug/runs/<new run>` (2x
   samples) and set `planner.speed_x2_fill_*`. Watch for "skipped: ... re-planning without it", "tower: ... cannot
   fill", "last-slot gamble: ... fills itself / opens ...", and whether the Kingshot-style end
   card is closed ("ad ✕ on a disc"). Open: `gamble_fill_margin` 1.0 vs 0.9 (#27 yellow).
1. Live L187 `--levels 3` (sand to the bottom edge changes every sand share: watch "not enough
   sand", the capacity line, and how ads end - frames every 10 s); after it `python -m tools.belt_speed`
   (adds the new run's samples if passed, and lists the plan checks) and re-tune `planner.fill_*`.
2. If Mirroring vanishes again: the `window list:` line says whether its window went
   off-screen (another Space / hidden) or was closed.
3. Hidden colours (under locks / curtains / ice) end the plan; with Try Again the colours seen in
   the last attempt could be filled in from memory (not done).
4. Belt reader: a cream lower stage on darkgreen (`darkgreen/cream`) may be a misread.
5. Answers on phase-5 issues A / B, then fixes A-D (see "Phase 5 state" below).

## Open gaps / questions
- **Hint arrows are missed in some animation phases** (template < 0.8; squashed arrows in
  run `125351` frames 00147 / 00154-156 even < 0.6). The lattice lock makes this harmless for
  the grid; a bottom-row cell under an unmasked arrow can read UNKNOWN ("two objects"), never
  tapped.
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
  `Analyzer.reset_level()` in bot.py / overlay; off for unrelated screenshots): never replaced
  mid-level, only unlocked by a non-gameplay screen or a new frame size. The column pitch must
  fit the walls inside the grid (`Lattice.fit`); a misfit reading never becomes the lock.
- Outside gameplay: `flow.py` picks the only allowed button (NEXT / PLAY / Quit / Try Again /
  ✕); never inside Keep Playing / Watch Ad / Get App / App Store boxes; every tap logged.
- Frame -> decision lives in `analyzer.py` (Analysis, Analyzer, DecisionGate); `strategy.py`
  is the greedy choice only.

# History (newest first)

## Run 20261021-112241 (2026-10-21): L189 WON, phantom L190 "win"
- See "Where it stands". Tests: `tests/test_run_20261021_112241.py`.

## L189 live run 20261021-101655 (2026-10-21): lost 3x, memory did not help
- See "Where it stands". Tests: `tests/test_run_20261021_101655.py`.

## L189 live run 20261021-092045 (2026-10-21): lost behind the ice, 200 s stuck
- See "Where it stands". Tests: `tests/test_run_20261021_092045.py`.

## L188 run 20261021-001541 + user recording (065-071), L189 start (072) - 2026-10-21
- See "Where it stands". Tests: `tests/test_l188_l189.py` (frames from the run, dataset shots
  065-069 / 072), annotated grids 065 / 066 / 072 in every grid test of `tests/test_vision.py`.

## L187 live run 20261021-000105 (2026-10-21): FIRST L187 WIN, win counted 3x
- See "Where it stands". Tests: `tests/test_run_20261021_000105.py` (frames 00260, 00268,
  00271, 00272).

## L187 live run 20261020-224631 (2026-10-20): endgame lost, ad end card ✕ missed
- See "Where it stands". Tests: `tests/test_run_20261020_224631.py` (frames 00427 end card,
  00313 App Store sheet, 00228 tap #24, 00262 tap #27); gamble tests in `tests/test_lookahead.py`
  updated to the fill rule.

## L187 live run 20261020-171133 (2026-10-20): endgame lost twice, ad stop
- 54 taps over 2 attempts; both ended 5/5 with buckets at 96-99 %, contact layer darkred 54 % /
  cream 40 %; plans late in attempt 2 said "3-5 dead on the belt at the end" and were followed.
- Ads after Try Again: 288 s black, 301 Trade Republic (skip ▶| top right), then the window was
  gone 1.2 s and a Sand Loop splash; level 2 started at 336 s. 617 s: Netflix, then black until
  the stop at 676 s (unknown_max_s 60).
  User: the Trade Republic ad is closed with the ▶| disc top right (what `adclose.find_skip`
  taps). Open: whether the user touched anything during the 1.2 s window loss / splash.
- Tests: `tests/test_run_20261020_171133.py` (frames 00535, 00551, 00301, 00587).

## L187 live run 20261020-161536 (2026-10-20): plan wait stalled 32 s, window lost
- 8 taps (6 by the start plan, 2 by the second re-plan), then 32 s "waits for the belt (k 3 > 2)"
  with yellow 36-37 % / orange 0 % / orange-cream 38 % unchanged (sand total 0.7516 constant from
  45 s on), then "no on-screen window owned by 'iPhone Mirroring'" at 76 s, stop at 86 s.
- Belt-speed data (3 runs): contact share < 5 %: median 0 %/s; 5-15 %: ~1 %/s; 20-25 %:
  14 %/s; 25-35 %: 26-30 %/s. The old simulator took up to 20 cells / lap from any exposed sand.
- Regression tests: `tests/test_run_20261020_161536.py` (frame 00060 + the belt per frame).

## L187 live run 20261020-145310 (2026-10-20): gamble worked, lost in the endgame
- 26 taps; #22 last-slot gamble cream r1c6 -> #23 purple r3c6: the sky drained. Then #24 orange
  r4c6 + burst #25 yellow r2c2 (0.15) and #26 purple r5c6 took the slots; belt 5/5 with colours
  at 64-95 % whose sand was gone (yellow 0 %, others ~1 %); darkred 39-50 % of the contact layer
  with 2 buckets on the board, rejected "not enough sand (1.0 % < 1.2 %)" - LEVEL FAILED (revive).
- #20 r5c0 purple (0.22) was a confirmed tap, not a burst: both darkred buckets were rejected for
  sand. #25 was a burst; it did not take the last slot (free 2 -> 1) but scored only 0.15.
- Tests: `tests/test_levelplan.py` with frames 00002 / 00134 / 00188 / 00201 and the run's
  levels.jsonl record in `tests/data/run-20261020-145310/`.

## L187 live run 20261020-142035 (2026-10-20): grid lock OK, belt deadlock - gamble + lookahead
- Lattice: 7x7 all run. 23 taps; #14-#23 had low scores; from 101 s the belt held yellow /
  darkgreen / orange / blue (+ darkred), all starving; "no bucket passes the rules" for 50 s.
- Why: the purple sky covered the bottom (contact layer purple 27 % at #21, 45 % at #23); the
  belt's colours were under it. The last purple buckets r3c6 / r5c6 sat behind cream r1c6, which
  became tappable after #17 (purple/cream tower r1c5). From then on cream was always rejected
  (last-slot 25 % at 61 s; "not enough sand 0.8 % < 1.3 %" at #21 with 3 free slots), and the
  planner treats a bucket that fails the rules as a wall, so no path to purple existed. The
  mistake: #21 orange r5c4 (22 %) + #22 blue r4c2 burst (22 %, blue sand 2 % < one bucket: it
  ended at 64 %) spent 2 of 3 free slots on colours buried under purple; better: cream r1c6
  (a lost slot) then purple r3c6 (+ orange r4c6, purple r5c6) to drain the sky. Earlier: #5
  yellow (burst, 10 % avail) sat on the belt from 16 s to the end and never filled.
- End: "belt idle" flips (k > buckets seen counts as collecting; orange crept 52 -> 80 % until
  112 s), so the last-slot minimum switched 25 % / 3 % and darkred r3c2 alternated between
  "exposure" and "not enough sand"; cream's avail -18.7 % = a cream lower stage the belt reader
  sees on darkgreen.
- Warnings "r3c2 darkred-up / r4c2 blue-up ... path rule says closed" (76.4 s): neither the path
  rule nor the colours - the bucket of tap #20 (r5c2) flying up the lane read as buckets in the
  empty cells r4c3 / r3c3 for one frame (also r1c3 at 20.8 s after #7), cutting the path; that
  frame even chose a phantom "tap r4c3 orange" (not confirmed). Fix: `hints.InFlight`.
- Offline replay: the end frame with the belt stuck now gambles darkred r3c2 (uncovers ~6 %
  orange / blue / darkgreen vs cream ~4 %); at #21 the lookahead marks blue dead (sand short).
- Tests: `tests/test_lookahead.py` (gamble rules, lookahead, InFlight, the run's frames in
  `tests/data/run-20261020-142035/`); `tests/test_bot.py` checks `levels.jsonl`.

## L187 live run 20261020-125351 (2026-10-20): 21 good taps, then 7x7 <-> 4x4 flip-flop — fixed
- First lock was 7x7 (frame 00002, fit 0.88) - correct. At 34 s one frame read 4x4 (sparkles
  of a finished bucket joined the grid at x 77). From 93 s, with the bot idle, the game's hint
  arrows over the boosters bobbed into the grid; the template missed them in some phases, the
  grid region grew to 68..531 px (was 101..497) and the lattice read 4x4 (cell 107 / 116 px).
  After 3 such frames LatticeLock took it ("new grid geometry"), then 7x7 again 3 frames later,
  back and forth. The 4x4 cells gave "two objects in one cell" unknowns
  (`dataset/unknown/20261020-1255*`) and the r1c1 "orange/cream-down looks face-down" warning
  (a merged cell, not the tower gap).
- Fixes: (1) `LatticeLock` never replaces the lock mid-level (see "Where it stands").
  (2) `Lattice.fit` = `vision.wall_alignment` of the chosen pitch: every grown-outline frame
  fits <= 0.39, every clean frame of 224 live / dataset gameplay frames >= 0.88 (only the
  dragged-window frame 122638/00058 is lower, 0.54, also a misread). A misfit never becomes the
  first lock; a finer reading replaces a coarse provisional lock. (3) open-tower foot (above).
- With the 7x7 lock the end frames still say "no bucket passes the rules" (belt deadlock, see
  NEXT 3) - that is strategy, not vision.
- Config: `lattice_switch_frames` -> `lattice_lock_frames` (3, refine window) +
  `lattice_min_fit` (0.8); `foot_min_obj` (0.5).
- Tests: `tests/data/run-20261020-125351/` (00002, 00047, 00137, 00143, 00147 + the two
  unknown-dir frames), `tests/test_run_20261020.py` (misfit frames, first lock, locked reads of
  the flip-flop frames), lock unit tests in `tests/test_run_20261019.py`.

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
