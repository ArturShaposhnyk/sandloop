"""Check the macOS permissions the bot needs and explain how to grant them.

    python -m permissions
"""
from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

_CG = "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
_AS = "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"


@dataclass
class PermissionStatus:
    name: str
    granted: Optional[bool]  # None = could not determine
    required: bool
    how_to: str


def _call_bool(lib_path: str, fn: str) -> Optional[bool]:
    try:
        f = getattr(ctypes.cdll.LoadLibrary(lib_path), fn)
    except (OSError, AttributeError):
        return None
    f.restype = ctypes.c_bool
    f.argtypes = []
    return bool(f())


def host_app() -> str:
    term = os.environ.get("TERM_PROGRAM", "")
    return {"Apple_Terminal": "Terminal", "iTerm.app": "iTerm", "vscode": "Visual Studio Code"}.get(
        term, term or "your terminal app")


def check() -> List[PermissionStatus]:
    app = host_app()
    pane = "System Settings → Privacy & Security → {} → enable “%s”, then fully quit and reopen %s." % (app, app)
    return [
        PermissionStatus("Screen Recording", _call_bool(_CG, "CGPreflightScreenCaptureAccess"), True,
                         pane.format("Screen & System Audio Recording")),
        PermissionStatus("Accessibility (clicks)", _call_bool(_AS, "AXIsProcessTrusted"), True,
                         pane.format("Accessibility")),
        PermissionStatus("Input Monitoring (hotkeys)", _call_bool(_CG, "CGPreflightListenEventAccess"),
                         False, pane.format("Input Monitoring")),
    ]


def request_screen_capture() -> None:
    """Ask macOS to show the Screen Recording prompt (adds the app to the list)."""
    _call_bool(_CG, "CGRequestScreenCaptureAccess")


def require(screen: bool = True, accessibility: bool = True) -> None:
    """Exit with an actionable message if a required permission is missing."""
    if sys.platform != "darwin":
        sys.exit("error: this bot needs macOS (iPhone Mirroring).")
    wanted = {"Screen Recording": screen, "Accessibility (clicks)": accessibility}
    missing = [s for s in check() if wanted.get(s.name) and s.granted is False]
    if missing:
        lines = ["error: missing macOS permission(s):"]
        lines += [f"  - {s.name}: {s.how_to}" for s in missing]
        sys.exit("\n".join(lines))


def main() -> int:
    if sys.platform != "darwin":
        print("This bot needs macOS (iPhone Mirroring).")
        return 1
    ok = True
    for s in check():
        mark = {True: "OK     ", False: "MISSING", None: "UNKNOWN"}[s.granted]
        tag = "" if s.required else " (optional)"
        print(f"[{mark}] {s.name}{tag}")
        if s.granted is not True:
            print(f"          {s.how_to}")
            ok = ok and not (s.required and s.granted is False)
    if not ok:
        request_screen_capture()
    print("\nAll required permissions granted." if ok else "\nFix the items above and re-run.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
