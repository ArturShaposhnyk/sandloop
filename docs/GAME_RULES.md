# Sand Loop — game rules (confirmed by the user)

## Buckets in the grid
- Face-up bucket (opening facing up) = playable, can be tapped.
- Face-down bucket (bottom facing up) = NOT tappable; its color is visible.
- **Path rule (playability):** a bucket is face-up / playable iff there is a path of EMPTY
  cells (4-neighbour) from it to the grid exit at the conveyor side. Front-row cells adjacent
  to the exit count as connected. Buckets, ice blocks, curtains (and UNKNOWN cells) are
  obstacles for the path. "Nothing directly in front" is NOT the rule — e.g. in shot 043 the
  red bucket at row 3 col 5 is open via the empty cell to its left, while the orange bucket at
  row 2 col 1 is closed because all its neighbours are occupied.
- The bot computes playability by BFS over the grid and also reads the visual face-up/down
  state; if they disagree it logs a warning (and does not tap).
- Ice block with a number N = locked; unlocks after N more buckets are tapped
  (placed on the conveyor or used in any way).
- Curtain with a number N = locked; unlocks after N more buckets are filled to 100%.
  Its size varies between levels (2x2 in L180 is just one example): always detect its extent
  from the gold frame on screen and map it to the covered grid cells.
  Under the curtain there are **buckets**, colours unknown until it opens. When it opens they
  become regular buckets and follow the path rule.
- **Locks and keys** (L183, confirmed by the user):
  - A **key** (yellow, red, blue (L187), ...) sits in its own grid cell. It is an obstacle for the path rule
    until a path opens to it.
  - When the key's cell connects to the exit by the path rule (e.g. the bucket in front of it is
    released), the key flies by itself to the **lock of the same colour** and opens it.
  - Under a lock there are **buckets** (not tappable while the lock is there). After it opens
    they become regular buckets under the path rule (colours unknown until seen).
  - They are playable **immediately**, while the padlock animation still covers them (user,
    2026-10-19; run 20261017-103804 r2c1, r3c5). They read face-down meanwhile, so for
    `vision.unlock_trust_s` after a key lock cell reads as a bucket / empty, the path rule
    alone decides there (`hints.RecentUnlocks`).
  - Bot rule (user, 2026-10-20): tap such a bucket at once **only if its colour is read
    confidently** - not the lock's own colour (the fading padlock reads as a bucket of the
    lock's colour), nearest colour within `vision.unlock_confident_dist` and the runner-up at
    least `vision.unlock_confident_margin` further. Otherwise wait until the padlock is gone
    (`vision.unlock_wait_s`, ~2 s), then the reading counts (`vision.unlock_unsure`).
  - After the key flies, its cell is **empty** (confirmed).
  - Several locks of the same colour: not seen yet (open question). Until then the bot credits
    no unlock when a key's colour has several separate locks.
  - Every lock in L183 covers one cell; adjacent lock cells of one colour are treated as one lock
    (not confirmed).
- **Chest** (L186, confirmed by the user): an orange wooden box with a number N, in its own cell.
  An obstacle for the path rule. It opens by itself when the path rule connects its cell to the
  exit (like a key) and scatters **N buckets** (colours from the level's palette, random) into
  free cells; buckets that were open can become closed because the new ones block paths.
  - Vision: chest-wood colour share (palette ui `chest_wood`) **and** a readable number; wood
    without a number is not a chest (a fading yellow padlock looks similar).
  - Lookahead: a tap that connects a chest opens it; worst case its cell stays an obstacle and
    its N buckets count as pending (unknown colour). A planned path ends at a step that opens a
    chest (the board after it is unknown).
  - After a chest opened (it is gone from the grid), the bot decides nothing for
    `strategy.chest_settle_s` and forgets old cell readings: the next decision is on a full
    re-read of the grid.
- **Mystery bucket** (L186, confirmed by the user): grey with "?", colour unknown. Revealed only
  when a bucket **directly 4-adjacent** to it is tapped (an original one or one a chest
  scattered); from then on it is a normal bucket under the path rule. Until then an obstacle,
  never tapped, counted as one pending bucket. Vision: grey share (palette ui `mystery_grey`);
  the "?" is not read as a counter.
- **Wall** (L186, confirmed by the user): a dark grey cell, never passable, never tappable. On
  screen it is a hole in the grid panel in the panel's own colour (LAB = `panel_bg`), so the
  lattice has no cell there (`x`), which blocks paths exactly like a wall.
- Tower bucket = stacked buckets in one cell (confirmed by the user, L183):
  - Tapping a tower sends the **whole tower** to the conveyor; its cell is **empty at once**
    (so the path rule, key release etc. apply immediately).
  - On the conveyor the tower takes **one slot** for all its stages. The top bucket collects
    first; when it is **100%** the next stage starts collecting (its colour), and so on
    ("clear the top bucket to unlock the next colorful stage", shot 030, happens on the belt).
  - On the belt a tower shows the collecting stage on top of the next stage's colour (L181:
    cream on red, red on orange); the fill % is the collecting stage's.
  - Seen in the grid from above: face-up = the top stage's opening; face-down = a disc of the
    top stage's colour on the dome of the stage below.
- Vision (L187): columns can come in groups separated by walls; the column pitch is the cell
  pitch, never a multiple of it (`vision.pitch_harmonic_ratio`; the walls inside the grid sit
  on cell boundaries, `vision.pitch_wall_align`). The grid geometry never changes within a
  level; a cell with a background row through its middle holds two objects (never a tower). Lock / key icons are matched
  on the cell's own rectangle (`vision.icon_inset`), not on the bucket crop shifted up into the
  row above.
- Colours seen so far: blue, cream, cyan, darkgreen, darkred, green, magenta, orange, pink, red,
  and from L187 **yellow** and **purple** (buckets and sand; purple = the sky there).
- Colour of an **open (face-up) bucket** = its **rim**: the inside is shaded darker, so a light
  green open bucket reads dark green from its inside (user, L183 shot 055 r4c0). The bot takes
  the brightest top-stage pixels (`vision.rim_percentile`) and the palette has a rim look per
  colour learned from the annotated open buckets. Close pairs (`vision.colour_watch`, e.g.
  green / darkgreen) are printed with the distance to both in the debug output.

## Levels are deterministic (user, 2026-10-20)
- **Try Again restarts the SAME level**: same buckets in the same cells, same colours.
- A face-down bucket's visible colour is its real colour, so the whole board is known at level
  start (except what locks / curtains / ice / chests / mystery buckets hide). Only **chests may
  be random** (not sure): the bot logs every chest result per attempt (`levels.jsonl` "chests":
  the cells and colours its buckets landed in) to check.

## Conveyor
- Max 5 buckets on the conveyor at once (HUD shows "k/5").
- A bucket rides the conveyor until it is 100% full, then disappears and frees its slot.
- "2x" toggle speeds up the conveyor — the bot may turn it on.

## Sand
- A bucket collects only sand of exactly its color (same shade).
- It collects only from the bottom of the picture (plus a few grains above).
- Sand falls with physics, so new sand reaches the bottom over time.
- Win: all sand collected, picture revealed.

## Lose condition
- All 5 conveyor slots are occupied by buckets that can't reach 100%
  (no more matching sand reachable) → level failed.

## Boosters
- 4 boosters at the bottom (bucket+, magnet, shuffle, +1 slot). The bot must NOT use them.

## Confirmed details
- A face-down bucket flips face-up as soon as the path rule connects it to the exit
  (see "Buckets in the grid").
- Bucket capacity is the same for all buckets within a level, but differs between levels
  (levels with more sand → buckets fill faster per pass).
  Working hypothesis: each level is balanced so that
  sand_of_color ≈ buckets_of_color × capacity. The strategy should reason in terms of
  "remaining sand of color C vs. remaining buckets of color C", not absolute capacity.
- A bucket collects matching sand along the whole picture width while riding the conveyor
  (bottom layer + a few grains above).
- A new bucket goes to the end of the queue / the next free slot on the conveyor.

## Strategy implications
- Never release a bucket unless enough of that color remains to fill it to 100%. How much of it
  must be reachable at the bottom right now depends on the belt (see "When to wait").
- **Reachable sand** ("exposed") = what a bucket can collect: per picture column the lowest
  ~28 px of sand (`vision.contact_layer`, empty picture skipped), not just the bottom row of
  grains - a bucket takes a few grains higher too (user, L183 shot 056: cyan sky right above
  the bottom read 2.6 % with a 10 px layer, ~15 % with the reach band).
- Playable buckets = buckets connected to the exit by the path rule (BFS), not locked.
- Tapping a bucket empties its cell, which can open a path to other buckets → lookahead must
  account for what gets opened.
- Prefer releasing colors whose sand is exposed at the bottom right now; avoid colors
  whose sand is buried under other colors (it will block a conveyor slot).
- Prefer moves that unlock face-down buckets, ice blocks, and curtains.
- Avoid deadlock: keep at least one conveyor slot usable.

## When to wait (user, 2026-10-16, L183 shot 056; thresholds in `config.yaml` `strategy:`)
- Waiting must have a reason: sand only moves while belt buckets are collecting.
- **Belt idle** (no unfinished bucket on the belt, or none of them has filled for `stall_s`):
  waiting changes nothing, so never wait - take the best legal move even with low exposure, as
  long as its colour has enough sand left overall. (Buckets the belt reader could not see,
  k > buckets read, count as collecting.)
- The exposure minimum scales with free slots: last free slot `last_slot_min_exposed` (25 %;
  `min_exposed` if the belt is idle); `relaxed_free_slots` (3) or more free:
  `relaxed_min_exposed` (0 = the colour need not be exposed, avail just not negative, i.e.
  not already claimed by same-colour belt buckets); otherwise `min_exposed` (3 %).
- A tap that opens a useful bucket, or is a step of a planned path, uses `opener_min_exposed`
  (0) once `opener_free_slots` (2) are free.
- **Stalled colour** (user, 2026-10-19): one more bucket of a colour that is stalled on the belt
  (e.g. cream at 99 % not moving) is a lost slot. While another belt bucket still collects it is
  never released (wait); on an idle belt it is a last resort with the `weight_stalled_colour`
  penalty, and a path step of that colour costs `path_cost_stalled` extra, so another move or
  path goes first. The stall clock ignores ±1 % label jitter (`stall_min_growth`) and frames
  where the colour is not read (`stall_forget_s`). The sufficiency rule always holds.

## Unlock value and paths (user, 2026-10-16; `planner.py`, weights in `config.yaml` `strategy:`)
- Face-down buckets show their colour, so what a move opens is plannable. The bot stays
  greedy + 1 tap of lookahead, plus a bonus for the first step of a planned path.
- **Usefulness of a colour** = max(share in the contact layer, `lower_weight` x share in the
  lower part of the picture (`vision.lower_band`)) - belt demand (`belt_demand_exposure` x what
  the same-colour belt buckets still need); 0 if not enough of its sand remains. Useful =
  `useful_min` and up. "Opens 2 buckets" is not enough: "opens 2 darkred buckets while darkred
  is at the bottom" is what scores.
- **Unlock value of a tap**: every bucket the tap connects to the exit (path rule; also after a
  key flies) is scored by its colour's usefulness in the state after the tap. Buckets under a
  curtain / lock / ice have no known colour: only the existing per-cell credits apply.
- **Path planning**: targets = useful buckets the path rule does not connect yet, and keys whose
  colour has exactly one lock (value `path_key_value` per bucket under it). For each, the
  cheapest set of buckets to remove that connects it to the exit (Dijkstra; empty cells cost 0,
  a bucket costs 1 + up to `path_cost_low_exposure` when its colour is barely exposed; locks,
  unknown cells and buckets not safe to release now are impassable).
- **Never a path that could deadlock the belt**: the path is replayed tap by tap and every step
  must pass the release rules in the state it would be tapped in (free slot, last-slot
  threshold, exposure minus the growing belt demand, enough sand, no tower stage short of sand).
  Conservative: it assumes no belt bucket fills up meanwhile.
- Best safe path: highest rank = value - `path_step_cost` x cost (must be > 0); its first step
  gets `weight_path_bonus` x rank. One-step paths are the tap's own unlock value (not planned).
  Plans are recomputed every frame (no commitment to a path yet).
- Overlay / report text: `r2c3 darkred -> opens r3c3 cream, r4c2 darkred (useful)`,
  `path to r5c1 darkred: step 1/3`, `path to key yellow: step 1/2` (rejected paths with the
  failing step and why).

## Last-slot gamble (user, 2026-10-20; `strategy.last_slot_gamble`)
- If the belt is **stuck** (no unfinished belt bucket gained sand for `strategy.gamble_stuck_s`,
  ~5 s), one slot is left and every remaining move is rejected by the last-slot exposure minimum,
  waiting changes nothing and the level is lost otherwise: take the last slot anyway.
- **Best** = the move whose colour, once collected, uncovers the most sand of the belt's colours
  (`sand.contact_layers`: per picture column the sand right above the contact layer);
  tie-break: the bucket's own exposed sand. Logged as "last-slot gamble".
- The 25 % rule stays for normal cases. A move short of sand may be gambled too (the level is
  lost otherwise; user-confirmed 2026-10-20), a colour stalled on the belt never.

## Sacrifice (user, 2026-10-20)
- A bucket that cannot fill now (not enough sand of its colour) may take a slot if the lookahead
  shows it opens the way to the colour that unblocks the belt (after 2-3 taps a belt colour that
  gets no sand now gets some) and at least one slot stays free after it (or it is the last-slot
  gamble). More of its colour may appear once the covering layer is collected, so it may still
  fill later. L187: cream r1c6 -> purple r3c6 drains the sky.

## Endgame (user, 2026-10-20, run 20261020-145310)
- With few colours left on the board (`strategy.endgame_colours`), keep a slot for each colour
  whose sand is still in the picture (>= `endgame_sand_min` of a bucket), that has buckets on the
  board and none on the belt, instead of adding a bucket of a colour that is almost done (on the
  belt already, or less than a bucket of sand left). Only a colour more exposed than the one
  being tapped is kept (no two colours waiting for each other).
- A burst follow-up passes the same checks as any tap (rules, lookahead) and needs
  `bot.burst_min_score`; it never takes the last free slot.
- **Key colours** (user, run 20261020-171133): a colour whose sand covers the last sand of belt
  buckets (belt colours < `look_min_exposed` of the contact layer while it covers them, or it
  fills `strategy.key_min_exposed` of the contact layer then) must stay reachable: keep the slots
  its nearest bucket needs (1 + the buckets in front of it on the path). Taps of key buckets or
  of buckets on that path are allowed; a key out of reach with today's free slots reserves
  nothing (`endgame.key_reserve`).
- A level plan that ends with dead belt buckets is worse than any with fewer (a dead bucket never
  leaves); a full belt of dead buckets = lost, never planned. A plan step that breaks the slot
  reserves above is not tapped (the strategy decides).

## Sand per bucket
- Bucket capacity is the same for all buckets of a level, but the picture reading under-counts
  some colours (L187 darkred, cream by 2-3x: darkred read 1 % of the picture while it was 39 % of
  the contact layer). The bot measures capacity **per colour** at level start (its sand / its
  bucket stages, the level being balanced per colour) and uses it for "enough sand left".
- Cause found in run 20261020-171133: the picture reaches ~12 px below the configured
  `vision.picture` zone, and the colours that settle at the bottom (darkred, cream) were cut off
  (L187 end: darkred 0.17 % in the zone, 1.3 % down to the edge). Sand is now read down to the
  picture's real bottom edge, and colours down to `vision.sand_min_share` (0.05 %) are reported.

## Lookahead (phase 7, `lookahead.py`; weights in `config.yaml` `strategy:`)
- Before each tap the bot simulates up to `look_depth` (3) taps: the candidate, then the
  `look_width` best-exposed follow-ups the release rules allow, each with `after_tap`.
- Sand model (one coarse step): the belt colours that collect now take their contact layer and
  the sand right above it falls down (what the belt uncovers, `sand.contact_layers`).
- A belt bucket is **live** if enough of its colour is left to fill what the belt needs of it
  (`look_sand_margin` x need x capacity) and its colour is exposed and not stalled, or the
  simulated collection uncovers enough of it. Value = (free slots + live buckets) / slots.
- Score += `weight_lookahead` x (best value over the continuations - value now). A tap that
  leaves the belt full with nothing live (a **jam**) is rejected unless the belt is idle anyway.
- Every level is recorded in `debug/levels.jsonl` (result, every tap with its decision: state,
  candidates, scores, lookahead; the strategy config) to tune the weights from results.

## Level plan (`levelplan.py`, `sim.py`; `config.yaml` `planner:`)
- At level start the bot searches tap / wait sequences (beam search) in a simplified simulator:
  the picture as per-column stacks of sand (buried colours included; collecting a colour lets
  the sand above fall), belt slots with laps, towers, keys / locks, ice, curtains; capacity per
  colour. It picks the sequence that never jams the belt (full belt, nothing moves) and collects
  the most sand. Buckets of unknown colour (opened locks, curtains, ice, chests) end the plan
  ("horizon"): once seen, it re-plans.
- Live: each frame's board is matched to the plan's predicted board before each tap; a match
  taps that step once the belt has no more buckets than the plan had there; no match = re-plan
  (at most every `planner.replan_min_s`). No step left: the greedy strategy decides.
- Level memory (`debug/levels.jsonl`): on Try Again (same board signature) the sequences of
  attempts that did not win are banned (the plan must deviate) and the capacity per colour is
  corrected from what those attempts really used.
- Belt speed (measured, not a game rule: `python -m tools.belt_speed` on belt.csv of runs
  142035 / 145310 / 161536, 458 samples): a belt bucket fills ~100 %/s x (its colour's share of
  the contact layer - 8 %), at most ~30 %/s; below ~10 % it gets almost nothing (L187 run
  161536: orange at 7 % of the contact layer stayed 0 % for 30 s). The simulator uses this
  (`planner.fill_*`); the bot needs ~4 s per tap (`planner.tap_s`).
- A plan's wait for the belt never stalls: given up when the belt is `planner.wait_grace_s`
  behind the plan's predicted time or no belt bucket gained sand for `planner.wait_stall_s`
  -> re-plan from the real belt; if that plan waits too, the greedy strategy decides until the
  board changes. Every planned tap logs "plan check": predicted vs real belt and time.

## Chests and mystery buckets (user, 2026-10-19, L186; weights in `config.yaml` `strategy:`)
- A move that opens a path to a chest is risky: its N buckets fill up to N free cells and can
  close paths. Prefer it when free cells are few, when it doesn't cut off important paths, or
  when nothing better is available. Bot heuristic (`planner.scatter_risk`): penalty =
  `weight_chest_cut` x expected number of "important" free cells hit (blocking one alone
  disconnects a bucket / key / lock now connected; N buckets land uniformly in the free cells)
  + `weight_chest_free` x share of the grid's cells that are free. A penalty, never a rule.
- Tapping a bucket next to a mystery bucket reveals its colour: small bonus
  `weight_mystery_reveal` (information) + `weight_mystery_reachable` when the revealed bucket is
  connected to the exit afterwards (with the path rule that is always the case right after the
  tap: the tapped cell becomes an empty cell on the exit side).

## Grid geometry (live run 20261019-162703, L186)
- The grid's shape is fixed within a level. The boosters below the grid can pulse into the grid
  zone when the player is idle; the grid region then grew by ~0.7 row and the lattice gained a
  row, shifting every cell (towers that are really two buckets, boosters read as buckets). The
  bot keeps the level's lattice (`hints.LatticeLock`): it is never replaced mid-level, only a
  non-gameplay screen (or another frame size) unlocks it; a reading whose inner walls miss its
  cell boundaries (`vision.lattice_min_fit`) never becomes the lock (L187 run 20261020-125351:
  hint arrows / sparkles joined the grid region and the lattice read 4x4).
- A tapped bucket flies up its lane over empty cells (~1.4 s): for `vision.flight_s` after a cell
  read empty, an object read there is that bucket and the cell stays empty (`hints.InFlight`;
  paused while a chest's buckets land). Run 20261020-142035: r3c3 / r4c3 read orange for one
  frame after tap #20 and cut the path of r3c2 / r4c2.
- If a cell's look and the path rule still disagree for `bot.disagree_report_s` while there is no
  other move, the bot logs STUCK and saves the frame (`stuck` event) instead of waiting silently.

## Unknown colours
- New levels bring new bucket / sand colours (L183: green and dark green grass). A bucket colour
  that matches nothing in the palette makes the cell UNKNOWN (never tapped); sand colours that
  match nothing are found as compact clusters of unmatched picture pixels. Both are logged,
  shown in the overlay (magenta) and saved once per run to `dataset/unknown_colours/`; add one
  with `python -m tools.add_colour <name> <crops>` (or annotate a screenshot and rebuild).

## Hint overlays (user, L183 shot 055)
- Big yellow **down arrows** over the bottom row of the grid point at the boosters. The game
  shows them when 4 buckets are on the belt and not full yet. They are a hint only: the bot
  ignores them (never a cell, a bucket or an unknown mechanic).
- Vision (`hints.py`): masked colour template `dataset/templates/hints/arrow_down.png`
  (`python -m hints cut <frame> x0 y0 x1 y1` for a new one). The arrow pixels are kept out of
  the grid outline and inpainted before cells and curtains are read. A cell up to
  `vision.hint_cover_max` hidden is read through the inpainted pixels; a cell hidden more than
  that, or one that reads UNKNOWN through them, takes its last clean reading from earlier
  frames (same lattice), else UNKNOWN "hidden by hint arrow". Such cells are never tapped and
  never saved to `dataset/unknown/`.

## Unknown / future obstacles
Later levels introduce new obstacle / lock types beyond ice and curtains. Vision and strategy
must be extensible:
- Every grid cell is classified as: **playable bucket** / **face-down bucket** /
  **known lock** (ice, curtain, …) / **UNKNOWN**.
- UNKNOWN cells are treated as locked: never tapped.
- When an UNKNOWN cell appears, save the full frame + a crop of that cell to `dataset/unknown/`
  and log it, so the user can explain the new mechanic and it gets added as a new lock type.
- Lock types are data-driven: one small class/entry per type (how to recognize it, how to read
  its counter, when it unlocks), not hardcoded if/else chains.

## Bot behaviour outside gameplay
- Win ("Well Done! / Level completed!"): tap **NEXT**. Never "Watch Ad" / "x2" (or the x1..x25
  multiplier).
- Home map: tap **PLAY**. Popups on it (shop offers, passes, events): close with the red **✕**
  only (it wins over PLAY when both are seen).
- Revive ("OUT OF SLOT" / "Keep Playing"): always press **Quit** (never spend coins); when Quit
  is not visible, the popup's red ✕ (shot 038): it does the same as Quit, LEVEL FAILED
  follows (user, 2026-10-20). Then **Try Again** on LEVEL
  FAILED. A revive / fail screen = the level is lost (counted once).
- Ads / popups: only tap ✕, and wait for it to appear. Never tap "Get App", purchases or
  "Activate". If an App Store sheet opens, close it (its ✕). "Take a break!" = wait.
  Ads are recognised by their anchors or the iOS status bar (the game hides it).
- Ads may have no known anchor at all (run 20261020-171133 after Try Again: Trade Republic
  "6 %" with a yellow "Download" button and two small yellow discs top left (mute) / top right
  (▶| skip); Netflix). An unknown screen within `flow.ad_after_tap_s` of a screen tap is
  searched for the ✕ and the **skip** (▶|) button - ✕ / skip only (user), never Download.
  The user closes the Trade Republic ad with the ▶| disc in the top right (confirmed).
  Unknown / ad screens are waited for `flow.unknown_max_s` / `ad_max_s` (3 min) with a frame
  saved every `flow.unknown_frame_s`.
- Stop the bot when lives reach 0 (heart in the top bar). PLAY / Try Again only with lives >= 1.
- Every tap outside gameplay needs `flow.confirm_frames` frames agreeing on the button, never
  lands inside a forbidden anchor (Keep Playing, Watch Ad, Get App, App Store badge) and is
  logged with its frame. Unknown screens / ads without ✕ / screens with nothing allowed that
  last too long stop the bot (frame saved), as do session limits (`config.yaml session:`).

## Open questions
- Path rule vs. picture: in shot 010 the blue bucket at r4c4 (0-based) looks face-down although
  the empty cell to its left connects it to the exit. Every other annotated bucket (7 grids)
  agrees with the path rule. Is there a delay before a bucket flips, or a nuance in the rule?
  (The bot logs it and does not tap such a bucket.)
