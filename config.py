"""Typed configuration loaded from config.yaml (pure Python, testable anywhere)."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Union

DEFAULT_PATH = Path(__file__).resolve().parent / "config.yaml"


class ConfigError(ValueError):
    pass


@dataclass
class WindowConfig:
    owner_name: str = "iPhone Mirroring"
    title_contains: Optional[str] = None
    app_bundle_id: str = "com.apple.ScreenContinuity"
    refresh_interval_s: float = 2.0


@dataclass
class ScreenConfig:
    capture: str = "window"
    content: Union[str, List[float]] = "auto"
    auto_max_trim: float = 0.15
    uniform_tolerance: float = 6.0


@dataclass
class ClicksConfig:
    backend: str = "quartz"
    press_duration_s: float = 0.05
    move_settle_s: float = 0.03
    max_taps_per_second: float = 4.0
    activate_before_tap: bool = True


@dataclass
class SafetyConfig:
    failsafe_corner_pt: float = 5.0
    hotkey_pause: str = "ctrl+alt+p"
    hotkey_quit: str = "ctrl+alt+q"


@dataclass
class StabilityConfig:
    threshold: float = 0.01
    timeout_s: float = 3.0
    interval_s: float = 0.1


@dataclass
class OverlayConfig:
    max_height: int = 900


@dataclass
class PathsConfig:
    dataset: str = "dataset"
    debug: str = "debug"
    palette: str = "dataset/palette.json"


@dataclass
class Config:
    window: WindowConfig = field(default_factory=WindowConfig)
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    clicks: ClicksConfig = field(default_factory=ClicksConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def _merge_section(section: Any, data: Any, name: str) -> Any:
    if data is None:
        return section
    if not isinstance(data, dict):
        raise ConfigError(f"[{name}] must be a mapping, got {type(data).__name__}")
    known = {f.name: f for f in dataclasses.fields(section)}
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(f"[{name}] unknown keys: {sorted(unknown)}")
    values = {}
    for key, value in data.items():
        default = getattr(section, key)
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise ConfigError(f"{name}.{key} must be true/false")
        elif isinstance(default, (int, float)) and not isinstance(default, bool):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"{name}.{key} must be a number")
            value = type(default)(value)
        values[key] = value
    return dataclasses.replace(section, **values)


def validate(cfg: Config) -> Config:
    content = cfg.screen.content
    if isinstance(content, str):
        if content != "auto":
            raise ConfigError('screen.content must be "auto" or [x, y, w, h]')
    else:
        if (not isinstance(content, (list, tuple)) or len(content) != 4
                or not all(isinstance(v, (int, float)) for v in content)):
            raise ConfigError("screen.content must be a list of 4 numbers [x, y, w, h]")
        x, y, w, h = (float(v) for v in content)
        if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 - x + 1e-9 and 0 < h <= 1 - y + 1e-9):
            raise ConfigError("screen.content must lie inside the window (relative 0..1)")
        cfg.screen.content = [x, y, w, h]
    if cfg.screen.capture not in ("window", "mss"):
        raise ConfigError("screen.capture must be 'window' or 'mss'")
    if cfg.clicks.backend not in ("quartz", "pyautogui"):
        raise ConfigError("clicks.backend must be 'quartz' or 'pyautogui'")
    if cfg.clicks.max_taps_per_second <= 0:
        raise ConfigError("clicks.max_taps_per_second must be > 0")
    if not 0 <= cfg.screen.auto_max_trim < 0.5:
        raise ConfigError("screen.auto_max_trim must be in [0, 0.5)")
    if cfg.window.refresh_interval_s <= 0:
        raise ConfigError("window.refresh_interval_s must be > 0")
    if not cfg.window.owner_name:
        raise ConfigError("window.owner_name must not be empty")
    return cfg


def from_dict(data: Optional[dict]) -> Config:
    cfg = Config()
    data = data or {}
    if not isinstance(data, dict):
        raise ConfigError("config root must be a mapping")
    unknown = set(data) - {f.name for f in dataclasses.fields(cfg)}
    if unknown:
        raise ConfigError(f"unknown config sections: {sorted(unknown)}")
    for f in dataclasses.fields(cfg):
        setattr(cfg, f.name, _merge_section(getattr(cfg, f.name), data.get(f.name), f.name))
    return validate(cfg)


def load_config(path: Union[str, Path, None] = None) -> Config:
    import yaml

    p = Path(path) if path else DEFAULT_PATH
    if not p.exists():
        if path:
            raise ConfigError(f"config file not found: {p}")
        return validate(Config())
    with p.open("r", encoding="utf-8") as fh:
        return from_dict(yaml.safe_load(fh))
