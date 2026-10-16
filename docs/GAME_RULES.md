# Sand Loop — game rules (confirmed by the user)

## Buckets in the grid
- Face-up bucket (opening facing up) = playable, can be tapped.
- Face-down bucket (bottom facing up) = NOT tappable; its color is visible.
- Ice block with a number N = locked; unlocks after N more buckets are tapped
  (placed on the conveyor or used in any way).
- Curtain with a number N = locked; unlocks after N more buckets are filled to 100%.

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
- Face-down bucket flips face-up when the bucket in front of it (closer to the conveyor)
  has been used, or when there is nothing in front of it. So in each column only the
  front-most bucket is playable; the rest unlock in order.
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
- Playable buckets = front bucket of each column, if face-up and not locked by ice/curtain.
- Tapping a bucket also reveals the next bucket in that column → lookahead must account
  for what gets revealed.
- Prefer releasing colors whose sand is exposed at the bottom right now; avoid colors
  whose sand is buried under other colors (it will block a conveyor slot).
- Prefer moves that unlock face-down buckets, ice blocks, and curtains.
- Avoid deadlock: keep at least one conveyor slot usable.
