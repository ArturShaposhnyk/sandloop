"""Load and validate config.yaml into typed dataclasses (pure, OS-independent)."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent / "config.yaml"

RelRect = Tuple[float, float, float, float]


class ConfigError(ValueError):
    pass


@dataclass
class ContentDetectConfig:
    border_tolerance: float = 18.0
    min_aspect: float = 0.40
    max_aspect: float = 0.56


@dataclass
class WindowConfig:
    owner_name: str = "iPhone Mirroring"
    title_contains: str = ""
    refresh_interval_s: float = 2.0
    content_rect: Optional[RelRect] = None  # None = auto-detect
    content_detect: ContentDetectConfig = field(default_factory=ContentDetectConfig)


@dataclass
class CaptureConfig:
    backend: str = "mss"


@dataclass
class StabilityConfig:
    threshold: float = 2.0
    consecutive: int = 2
    timeout_s: float = 3.0
    poll_s: float = 0.08
    downscale_width: int = 160


@dataclass
class ClickConfig:
    backend: str = "quartz"
    press_s: float = 0.05
    activate_before_click: bool = True
    activate_settle_s: float = 0.3


@dataclass
class SafetyConfig:
    dry_run: bool = False
    max_taps_per_sec: float = 3.0
    failsafe_corner_pt: float = 5.0
    hotkey_pause: str = "<ctrl>+<alt>+p"
    hotkey_quit: str = "<ctrl>+<alt>+q"


@dataclass
class OverlayConfig:
    max_height: int = 900
    grid_step: float = 0.1


@dataclass
class PathsConfig:
    dataset: str = "dataset"
    debug: str = "debug"
    palette: str = "dataset/palette.json"


@dataclass
class Config:
    window: WindowConfig = field(default_factory=WindowConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    click: ClickConfig = field(default_factory=ClickConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


CAPTURE_BACKENDS = ("mss", "quartz")
CLICK_BACKENDS = ("quartz", "pyautogui")


def _parse_content_rect(value: Any) -> Optional[RelRect]:
    if value is None or (isinstance(value, str) and value.strip().lower() == "auto"):
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ConfigError("window.content_rect must be 'auto' or [x, y, w, h]")
    x, y, w, h = (float(v) for v in value)
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > 1.0001 or y + h > 1.0001:
        raise ConfigError(f"window.content_rect {value} must lie within 0..1 of the window")
    return (x, y, w, h)


def _build(cls: type, data: Any, path: str) -> Any:
    """Recursively build dataclass `cls` from a dict, rejecting unknown keys."""
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"'{path}' must be a mapping")
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(f"unknown key(s) in '{path}': {sorted(unknown)}")
    kwargs: dict = {}
    defaults = cls()
    for name, value in data.items():
        key_path = f"{path}.{name}" if path else name
        default = getattr(defaults, name)
        if name == "content_rect" and cls is WindowConfig:
            kwargs[name] = _parse_content_rect(value)
        elif is_dataclass(default):
            kwargs[name] = _build(type(default), value, key_path)
        elif isinstance(default, bool):
            if not isinstance(value, bool):
                raise ConfigError(f"'{key_path}' must be true/false")
            kwargs[name] = value
        elif isinstance(default, (int, float)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"'{key_path}' must be a number")
            kwargs[name] = type(default)(value)
        else:
            kwargs[name] = "" if value is None else str(value)
    return cls(**kwargs)


def _validate(cfg: Config) -> None:
    if cfg.capture.backend not in CAPTURE_BACKENDS:
        raise ConfigError(f"capture.backend must be one of {CAPTURE_BACKENDS}")
    if cfg.click.backend not in CLICK_BACKENDS:
        raise ConfigError(f"click.backend must be one of {CLICK_BACKENDS}")
    if cfg.safety.max_taps_per_sec <= 0:
        raise ConfigError("safety.max_taps_per_sec must be > 0")
    if cfg.window.refresh_interval_s <= 0:
        raise ConfigError("window.refresh_interval_s must be > 0")
    if not cfg.window.owner_name:
        raise ConfigError("window.owner_name must not be empty")
    if cfg.stability.consecutive < 1 or cfg.stability.downscale_width < 8:
        raise ConfigError("stability.consecutive >= 1 and stability.downscale_width >= 8")


def config_from_dict(data: Optional[dict]) -> Config:
    cfg = _build(Config, data or {}, "")
    _validate(cfg)
    return cfg


def load_config(path: Optional[Path | str] = None) -> Config:
    p = Path(path) if path else DEFAULT_PATH
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return config_from_dict(data)
