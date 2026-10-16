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
- Tower bucket = stacked buckets in one cell. Only the top stage is tappable; the next stage
  unlocks only when the top bucket is **filled to 100%** ("clear the top bucket to unlock the
  next colorful stage", shot 030).

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
- Never release a bucket unless matching sand is currently reachable at the bottom
  AND enough of that color remains to fill it to 100%.
- Playable buckets = buckets connected to the exit by the path rule (BFS), not locked.
- Tapping a bucket empties its cell, which can open a path to other buckets → lookahead must
  account for what gets opened.
- Prefer releasing colors whose sand is exposed at the bottom right now; avoid colors
  whose sand is buried under other colors (it will block a conveyor slot).
- Prefer moves that unlock face-down buckets, ice blocks, and curtains.
- Avoid deadlock: keep at least one conveyor slot usable.

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
- Revive ("OUT OF SLOT" / "Keep Playing"): always press **Quit** (never spend coins), then
  **Try Again** on LEVEL FAILED.
- Ads / popups: only tap ✕, and wait for it to appear. Never tap "Get App", purchases or
  "Activate". If an App Store sheet opens, close it.
- Stop the bot when lives reach 0.

## Open questions
- Win screen (level_complete): not captured yet.
- Path rule vs. picture: in shot 010 the blue bucket at r4c4 (0-based) looks face-down although
  the empty cell to its left connects it to the exit. Every other annotated bucket (7 grids)
  agrees with the path rule. Is there a delay before a bucket flips, or a nuance in the rule?
  (The bot logs it and does not tap such a bucket.)
