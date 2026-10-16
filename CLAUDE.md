# CLAUDE.md — Sand Loop bot

Full project brief: see `PROMPT.md`. These are the standing rules for every session.

## Hard constraints
- The bot interacts with the game **only** via screenshots of the iPhone Mirroring window and mouse
  clicks into it. No game modification, network interception, or memory access.
- Target platform is macOS (Quartz, iPhone Mirroring). Keep macOS-only imports lazy so pure logic
  and tests run on any OS.

## Conventions
- Python 3.11+, type hints everywhere, small modules (see layout in `PROMPT.md`).
- All game logic uses **relative coordinates 0..1 of the iPhone screen area**. Pixel/point
  conversion lives only in `capture.py` / `actions.py`.
- Tunables (thresholds, delays, zones, palette path, click backend, window owner name) go in
  `config.yaml`, not in code.
- Colors are classified in HSV/LAB against a palette built from `dataset/`, never hardcoded RGB.
- Tests: `pytest`, run on saved screenshots — never require the live game.

## Safety (never remove)
- Failsafe (cursor in a screen corner = stop), pause/quit hotkeys, `--dry-run`, taps-per-second
  limit, no clicks outside the detected iPhone screen rect.

## Workflow
- Work phase by phase (see `PROMPT.md`). After each phase: stop, show what was done and how to
  verify it on the Mac, and wait for the user's OK.
- Don't invent game mechanics — ask the user.
