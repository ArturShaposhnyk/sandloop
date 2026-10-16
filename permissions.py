"""Check (and optionally request) the macOS permissions the bot needs.

  python -m permissions            # check, print actionable fixes, exit 1 if anything is missing
  python -m permissions --request  # also trigger the system prompts

Permissions are granted to the *terminal app* that runs Python (Terminal, iTerm2, VS Code, ...),
not to Python itself. After toggling them, quit and reopen that terminal app.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

SETTINGS = "System Settings → Privacy & Security"


@dataclass
class Check:
    name: str
    ok: Optional[bool]  # None = could not determine
    fix: str


def _host_app() -> str:
    return os.environ.get("TERM_PROGRAM") or os.environ.get("__CFBundleIdentifier") or "your terminal app"


def check_screen_recording(request: bool = False) -> Check:
    fix = (f"{SETTINGS} → Screen & System Audio Recording → enable '{_host_app()}', "
           "then quit and reopen it.")
    try:
        import Quartz
    except ImportError:
        return Check("Screen Recording", None, "pyobjc-framework-Quartz is not installed")
    ok = bool(Quartz.CGPreflightScreenCaptureAccess())
    if not ok and request:
        Quartz.CGRequestScreenCaptureAccess()
    return Check("Screen Recording", ok, fix)


def check_accessibility(request: bool = False) -> Check:
    fix = (f"{SETTINGS} → Accessibility → enable '{_host_app()}' (needed for clicks and window "
           "activation), then quit and reopen it.")
    try:
        import ApplicationServices as AS
    except ImportError:
        return Check("Accessibility", None, "pyobjc-framework-ApplicationServices is not installed")
    if request:
        opts = {AS.kAXTrustedCheckOptionPrompt: True}
        ok = bool(AS.AXIsProcessTrustedWithOptions(opts))
    else:
        ok = bool(AS.AXIsProcessTrusted())
    return Check("Accessibility", ok, fix)


def check_window_titles_visible() -> Check:
    """Secondary signal: without Screen Recording, CGWindowList hides all window titles."""
    fix = "Window titles are hidden — a sign Screen Recording is not effective yet (reopen the terminal)."
    try:
        from capture import list_windows
        wins = list_windows()
    except Exception as e:
        return Check("Window titles visible", None, f"window list failed: {e}")
    if not wins:
        return Check("Window titles visible", None, "no windows on screen")
    return Check("Window titles visible", any(w.title for w in wins), fix)


def input_monitoring_note() -> str:
    return (f"Global hotkeys (pynput) may also need {SETTINGS} → Input Monitoring → "
            f"'{_host_app()}'. If hotkeys don't react, enable it there.")


def run_checks(request: bool = False) -> List[Check]:
    return [check_screen_recording(request), check_accessibility(request), check_window_titles_visible()]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m permissions", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--request", action="store_true", help="trigger the system permission prompts")
    args = ap.parse_args(argv)
    if sys.platform != "darwin":
        print("This bot only runs on macOS.")
        return 1
    checks = run_checks(args.request)
    missing = False
    for c in checks:
        mark = {True: "OK     ", False: "MISSING", None: "UNKNOWN"}[c.ok]
        print(f"[{mark}] {c.name}")
        if c.ok is not True:
            print(f"          → {c.fix}")
            missing = missing or c.ok is False
    print(f"\nNote: {input_monitoring_note()}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
