# Screen labels

One line per screenshot. Types: gameplay / level_complete / fail / revive / ad / menu / other.
Collected 2026-10-16 with `python -m tools.collect` (iPhone screen area, 602x1302 px, scale 2).

level_complete: shots 059-063 (L186 win, 2026-10-19; after winning L180 the game had gone straight
to an ad break). Their capture session numbered them 001-005, renamed to 059-063 (unique seqs).

| file | type | note |
|---|---|---|
| 20261016-102033-862-001-auto.png | gameplay | L180 start, 0/5, full picture, ice 17/12/15/20, curtain 12 |
| 20261016-102106-103-003-auto.png | gameplay | L180, first buckets on the conveyor, sand starts to fall |
| 20261016-102117-908-004-auto.png | gameplay | L180, 3/5, fill % under conveyor buckets |
| 20261016-102139-037-006-auto.png | gameplay | L180, 5/5 (conveyor full), picture collapsed to a pile |
| 20261016-102213-911-008-auto.png | gameplay | L180, 3/5, curtain 1, few buckets left |
| 20261016-102234-122-010-auto.png | gameplay | L180, 3/5, almost empty grid |
| 20261016-102247-831-011-auto.png | gameplay | L180, empty grid, 5/5, last sand being collected |
| 20261016-102305-448-012-auto.png | ad | "Take a break!" interstitial before an ad, over gameplay |
| 20261016-102310-385-013-auto.png | ad | video ad (no close button yet), mute + Get App |
| 20261016-102327-385-018-auto.png | ad | App Store sheet opened by the ad, close (x) bottom right |
| 20261016-102407-977-019-auto.png | ad | playable ad end card, Get App |
| 20261016-102409-173-020-auto.png | ad | ad end card without Get App banner (recognised by the iOS status bar) |
| 20261016-102415-198-021-auto.png | other | black frame (transition / ad closing) |
| 20261016-102416-612-022-auto.png | other | Sand Loop splash / loading |
| 20261016-102418-506-023-auto.png | menu | home map, level 181, PLAY button, 2/30 stars |
| 20261016-102425-114-024-auto.png | menu | popup: Tidal Offer (shop), close (x) top right |
| 20261016-102427-476-025-auto.png | menu | popup: Sandy Pass rewards track, close (x) |
| 20261016-102428-662-026-auto.png | menu | popup: Sandy Pass purchase 11,99 USD, close (x) |
| 20261016-102434-135-028-auto.png | menu | home map, level 181, 9/30 stars |
| 20261016-102440-043-030-auto.png | gameplay | L181 start with tutorial hint; NEW obstacle: stacked tower bucket |
| 20261016-102442-923-031-auto.png | gameplay | L181, cross-shaped grid, 0/5 |
| 20261016-102445-075-032-auto.png | gameplay | L181, 2/5 |
| 20261016-102449-582-033-auto.png | gameplay | L181, 5/5 (conveyor full) |
| 20261016-102518-909-035-auto.png | gameplay | L181, 4/5, picture partly collected |
| 20261016-102521-772-036-auto.png | revive | conveyor full, Keep Playing (900 coins) / Quit |
| 20261016-102528-468-037-auto.png | revive | OUT OF SLOT popup, Keep Playing 900 |
| 20261016-102529-880-038-auto.png | revive | OUT OF SLOT with close (x) + special offer 5,99 USD |
| 20261016-102536-802-039-auto.png | fail | LEVEL FAILED, Try Again |
| 20261016-102541-796-041-auto.png | gameplay | L181 retry start, lives 4 with timer |
| 20261016-102548-249-043-auto.png | gameplay | L181 retry, 4/5 |
| 20261016-102557-076-045-auto.png | gameplay | L181 retry, 5/5, one bucket at 100% |
| 20261016-102620-667-051-auto.png | menu | popup: Sandy Race event, START, close (x) |
| 20261016-113331-000-052-live.png | gameplay | L181, 1/5, red 49% stalled on the belt, contact layer darkred/orange: regression for demand-aware exposure (expect darkred) |
| 20261016-165727-433-053-live.png | gameplay | L183 start, 0/5: new colours green + darkgreen (grass), curtains 10/15/20, yellow + red locks and keys, green/darkgreen tower tops |
| 20261016-173352-327-054-live.png | gameplay | L183, 2/5 (cyan 79%, cream 33%): face-up darkgreen/cyan tower r2c5 next to the yellow key was read face-down (regression: expect tap r2c5 darkgreen) |
| 20261016-181207-171-055-live.png | gameplay | L183, 4/5 (cyan 90%, orange 25%, cyan 13%): yellow hint arrows over the bottom row point at the boosters (regression: mask them; r4c0 is an open LIGHT green bucket, expect tap r4c0 green) |
| 20261016-185258-702-056-live.png | gameplay | L183, 1/5 (cream 33%): only r1c1 cyan tappable, opens a path; 10 px contact layer read cyan 2.6 % and the bot waited (regression: reach band + no pointless wait, expect tap r1c1 cyan) |
| 20261019-162708-850-057-live.png | gameplay | L186 start, 1/5 (cream 68%): new cell types - chests "4" at r4c1 / r4c5 (open when connected, scatter 4 buckets), mystery buckets "?" at r5c1 / r5c5 (revealed by tapping a neighbour), wall at r3c3 (panel-coloured hole: 'x') |
| 20261019-162746-400-058-live.png | gameplay | L186, 4/5 (orange 26%, red 21%, cream 69%, pink 35%): the frame the live run froze on - every 2nd frame the boosters below joined the grid region, the lattice got a 7th row and every cell was read across two rows (regression: 6 rows; with the belt idle expect tap r4c0 darkgreen) |
| 20261019-171450-497-059-auto.png | level_complete | L186 won: "Well Done! / Level completed!", NEXT (right), Watch Ad x2 (left, never) |
| 20261019-171509-276-060-snap.png | level_complete | same screen |
| 20261019-171510-884-061-snap.png | level_complete | same screen |
| 20261019-171512-124-062-auto.png | level_complete | same screen |
| 20261019-171515-475-063-auto.png | level_complete | same screen, confetti, 27 % |
| 20261020-120606-000-064-live.png | gameplay | L187 (HARD) start, 0/5, dry run 20261020-120601: new colours yellow + purple (buckets and sand, purple sky 28 %), yellow + BLUE locks and keys, walls every two columns (the 2x column pitch won the lattice: 4x4 instead of 7x7; regression) |
