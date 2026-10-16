# Sand Loop bot

Auto-player for *Sand Loop* (Voodoo), played through **iPhone Mirroring** on macOS. The bot sees the
game only through screenshots of the mirroring window and acts only by clicking into it.

Status: phases 0–1 done (setup, capture, taps); phase 2 (dataset collection) in progress. See `PROMPT.md` for the full plan.

## Setup

```bash
python3 -m venv .venv            # Python 3.11+ recommended; 3.9+ works
source .venv/bin/activate
pip install -r requirements.txt
python -m permissions            # checks Screen Recording / Accessibility / Input Monitoring
```

Permissions belong to the app that runs Python (Terminal, iTerm, VS Code…):
System Settings → Privacy & Security →

- **Screen & System Audio Recording**: required (screenshots, window titles)
- **Accessibility**: required (clicks)
- **Input Monitoring**: optional (global pause/quit hotkeys)

After you toggle a permission, fully quit the terminal app and reopen it.

## Commands

| Command | What it does |
|---|---|
| `python -m capture list` | All on-screen windows: id, pid, layer, bounds, owner, title |
| `python -m capture info` | Resolved iPhone Mirroring window, detected screen rect, scale factor |
| `python -m capture shot` | Saves `debug/window-*.png` (detected screen in green) and `debug/screen-*.png` |
| `python -m tools.overlay [--window]` | Live view with FPS, frame diff, and the relative coord under the cursor |
| `python -m actions tap 0.5 0.8 [--dry-run] [--backend pyautogui] [--count N]` | Tap a relative point |
| `python -m tools.collect [--manual] [--label win] [--max 40]` | Save dataset screenshots to `dataset/screens/` (auto on each new settled screen; `ctrl+alt+s` = snap now) |
| `pytest` | Unit tests (no macOS or live game needed) |

## Coordinates

All game logic uses coordinates **relative to the iPhone screen area (0..1)**. Only `capture.py` and
`actions.py` convert them to capture pixels (×scale, detected at runtime) or global screen points.

If the app name is localized, find it with `python -m capture list` and set `window.owner_name`.
If auto-detection of the screen area is off, set `screen.content: [x, y, w, h]` (relative to the
window) in `config.yaml`.

## Safety

- Moving the cursor into any screen corner stops tapping (checked before and after the rate-limit
  wait; the pyautogui backend also has its own `FAILSAFE`).
- Hotkeys (configurable): `ctrl+alt+p` pause/resume, `ctrl+alt+q` quit. Ctrl+C always works.
- `--dry-run` prints targets and never clicks.
- Taps are rate-limited (`clicks.max_taps_per_second`) and refused outside the iPhone screen rect.
