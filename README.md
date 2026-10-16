# Sand Loop bot

Auto-player for *Sand Loop* (Voodoo) played through **iPhone Mirroring** on macOS.
It only **sees** the game via screenshots of the mirroring window and only **acts** via mouse
clicks into it (a click = a tap). See `PROMPT.md` for the full brief and phase plan.

Status: **phases 0–1** — project setup, capture, taps, live overlay.

## Setup

Python 3.11+ is recommended (the code also runs on the macOS system Python 3.9).

```bash
# Python 3.11+: python.org installer, or `curl -LsSf https://astral.sh/uv/install.sh | sh && uv venv -p 3.12 .venv`
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Permissions (granted to your *terminal app*, not to Python)

System Settings → Privacy & Security:

| Permission | Why |
|---|---|
| **Screen & System Audio Recording** | capture the iPhone Mirroring window |
| **Accessibility** | send clicks, bring the window to front |
| **Input Monitoring** (if hotkeys don't react) | global pause/quit hotkeys |

After enabling, **quit and reopen** the terminal app. Then check:

```bash
python -m permissions            # exit 0 = all good; prints exact fixes otherwise
python -m permissions --request  # also triggers the system prompts
```

## Phase 1 — verify on your Mac

1. Open iPhone Mirroring and start Sand Loop.
2. Find the window:
   ```bash
   python -m capture list    # copy the OWNER of the mirroring window if it isn't "iPhone Mirroring"
   python -m capture info    # window bounds, scale factor, detected iPhone screen rect
   python -m capture shot    # saves debug/shot.png — check that it is exactly the phone screen
   ```
   If the crop includes the title bar / frame or cuts the screen, set
   `window.content_rect: [x, y, w, h]` (fractions of the window) in `config.yaml`.
3. Live view: `python -m tools.overlay` — frame + FPS + a 0.1 grid; hover to read `rel (rx, ry)`.
   Keys: `q` quit, `s` save frame, `g` grid. With the default `mss` backend keep the overlay
   *beside* the mirroring window (or use `--backend quartz`).
4. Tap test (pick a harmless spot first):
   ```bash
   python -m actions tap 0.5 0.5 --dry-run           # prints the target point, no click
   python -m actions tap 0.5 0.5                     # Quartz CGEvent click
   python -m actions tap 0.5 0.5 --backend pyautogui # fallback if Quartz clicks are ignored
   ```
   If only `pyautogui` works, set `click.backend: pyautogui` in `config.yaml`.

## Safety

- **Failsafe:** move the cursor into any screen corner → stops before the next tap (both backends).
- **Hotkeys** (long-running commands, e.g. `actions tap ... --repeat N`): `ctrl+alt+p` pause/resume,
  `ctrl+alt+q` quit (configurable).
- `--dry-run` / `safety.dry_run`: never clicks.
- `safety.max_taps_per_sec` rate limit.
- Taps outside 0..1 or outside the detected iPhone screen rect are rejected.

## Layout

| File | Role |
|---|---|
| `config.yaml`, `config.py` | tunables + typed, validated loader |
| `geometry.py` | the only relative ↔ pixel ↔ point conversion (window rect + content rect + scale) |
| `capture.py` | window lookup (CGWindowList), mss/quartz grabbing, scale + screen-rect detection |
| `actions.py` | `Tapper`: all safety checks + Quartz/pyautogui clicks + window activation |
| `safety.py` | rate limiter, corner failsafe, global hotkeys |
| `stability.py` | wait until the animated frame settles |
| `permissions.py` | permission check |
| `tools/overlay.py` | live debug view |

## Tests

```bash
pytest -q   # pure logic only — no live game, no permissions needed
```
