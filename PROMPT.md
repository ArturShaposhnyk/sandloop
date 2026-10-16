# Sand Loop auto-player — kickoff prompt for Claude Code

I want to build an auto-player bot for the mobile game **Sand Loop** (by Voodoo), running on my iPhone.
I play it through **iPhone Mirroring on a Mac** (macOS, works fine). The bot must **see the game only
through screenshots of the iPhone Mirroring window** and **control it only with mouse clicks into that
window** (a click = a tap on the iPhone). No game modification, no network interception, no memory
reading — screen + taps only.

## The game

A puzzle: a conveyor belt moves along the bottom / around a loop, and you tap to release buckets of
certain colors onto it. A bucket rides the conveyor and collects sand of its own color from a sand
picture. The goal is to collect all the sand and reveal the picture. Sand falls with physics. There
are limits (number of places on the conveyor / bucket slots), obstacles, and hard levels.

**You don't know the exact rules yet.** First study the screenshots I will provide, then ask me
clarifying questions about the mechanics **before** writing any strategy.

## Stack

Python 3.11+, `opencv-python`, `numpy`, `mss` (screenshots), `pyobjc-framework-Quartz` (window lookup,
clicks via CGEvent), `pyautogui` as a fallback for clicks, `pyyaml` for config, `pytest`.

## Layout

```
sandloop-bot/
  config.yaml          # thresholds, colors, delays, zone coordinates (relative 0..1)
  capture.py           # find the iPhone Mirroring window (Quartz CGWindowList), grab via mss
  actions.py           # tap(rx, ry) in window-relative coordinates; window activation
  vision.py            # detect conveyor, buckets, sand colors, UI buttons
  screens.py           # screen classification: gameplay / level_complete / fail / revive / ad / menu
  state.py             # dataclass model of the level state
  strategy.py          # choose the next move: greedy first, lookahead later
  bot.py               # main loop
  tools/collect.py     # save screenshots into dataset/ on a hotkey or every N seconds
  tools/calibrate.py   # show a frame, click zones with the mouse → write relative coords to config
  tools/overlay.py     # live debug view: frame + drawn detections + next move
  dataset/  debug/  tests/
```

## Technical details (important)

- **Window lookup:** use `CGWindowListCopyWindowInfo` and match by owner name `"iPhone Mirroring"`.
  Account for a localized app name — provide a `list` command that prints all on-screen windows
  (owner, title, id, bounds) so I can pick the right one; make the owner name configurable.
  Re-resolve the window bounds every ~2 s because the window can be moved.
- **Retina:** `mss` returns physical pixels (usually ×2), while clicks are in points. All internal
  logic works in **relative coordinates 0..1 of the iPhone screen area**; conversion to pixels/points
  happens **only** in `capture` / `actions`. Detect the scale factor at runtime, don't hardcode ×2.
- **Window chrome:** account for the window frame / title bar — detect the actual iPhone screen area
  inside the window (e.g. by finding the rounded-rect content region), and allow a manual override
  in config.
- **Clicks:** try Quartz `CGEvent` first (mouseDown / mouseUp with a ~50 ms pause between). If iPhone
  Mirroring ignores them, fall back to `pyautogui`. Make the backend selectable in config/CLI.
  Activate the window via `osascript` before the first click (and re-activate if focus was lost).
- **Permissions:** the terminal needs **Screen Recording** and **Accessibility**. Check both at
  startup (e.g. `CGPreflightScreenCaptureAccess`, `AXIsProcessTrusted`, or detect an all-black /
  title-less capture) and print a clear, actionable error saying exactly where to enable them.
- **Safety:**
  - `pyautogui.FAILSAFE` (mouse to a screen corner = stop), and an equivalent check for the Quartz
    backend (stop if the cursor is in a corner).
  - A global hotkey to pause/resume, and one to quit.
  - `--dry-run` mode: only draws decisions in the overlay, never clicks.
  - A rate limit on taps per second (configurable).
  - Never click outside the detected iPhone screen rect.
- **Frame stability:** before every decision wait until the frame stabilizes (frame-to-frame
  difference below a threshold, with a timeout), because the sand is animated.
- **Colors:** classify colors in HSV/LAB against a **palette collected from the dataset**, not
  hardcoded RGB values. Provide a tool/step to build and save the palette.

## Phased plan (after each phase — stop, show the result, wait for my OK)

0. **Project setup:** `requirements.txt`, `config.yaml` skeleton, README with setup instructions,
   permission check command.
1. **capture + actions:** `python -m tools.overlay` shows a live frame with FPS; a CLI command taps a
   given relative point (e.g. `python -m actions tap 0.5 0.8`). I verify that the tap reaches the game.
2. **`tools/collect.py`:** I'll collect 20–40 screenshots of different levels and states (start,
   mid-level, win, fail, ad). You analyze them, describe the UI layout and the mechanics to me, and
   ask questions.
3. **`vision.py` + `screens.py`** with tests on the saved screenshots (pytest, no live game needed).
4. **`state.py` + `strategy.py` (greedy)**, dry-run mode with overlay.
5. **`bot.py`:** full loop on a single level; log every move to `debug/` (frame + decision).
6. **Non-gameplay screens:** next level, retry, close ads when the ✕ appears.
7. **Strategy improvements:** simulate move consequences and search 2–3 moves ahead; metric — % of
   levels passed on the first attempt.

**Start with phases 0–1.** Write modular code with type hints.

### Notes for working in this repo

- The code runs on **macOS only** (Quartz / iPhone Mirroring). If you are in a non-macOS environment,
  write the code so that imports of macOS-only modules are lazy and the pure logic (coordinate
  conversion, config loading, frame-stability, color classification) is unit-testable anywhere.
- Keep coordinate conversion in one place and cover it with tests (window rect + content rect +
  scale factor → relative ↔ pixel ↔ point).
- Don't guess game mechanics: when something is unclear, ask me.
