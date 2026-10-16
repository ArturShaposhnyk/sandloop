"""Typed configuration loaded from config.yaml (pure Python, testable anywhere)."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
    hotkey_save: str = "ctrl+alt+s"   # global: save a report (frame + decision) to debug/


@dataclass
class CollectConfig:
    hotkey_snap: str = "ctrl+alt+s"
    change_threshold: float = 0.03
    min_interval_s: float = 1.0
    poll_s: float = 0.2


@dataclass
class VisionConfig:
    # zones, relative [x0, y0, x1, y1] of the iPhone screen
    picture: List[float] = field(default_factory=lambda: [0.18, 0.125, 0.82, 0.425])
    conveyor: List[float] = field(default_factory=lambda: [0.06, 0.448, 0.94, 0.49])
    fill_text: List[float] = field(default_factory=lambda: [0.0, 0.485, 1.0, 0.512])
    counter: List[float] = field(default_factory=lambda: [0.42, 0.505, 0.58, 0.548])
    grid_zone: List[float] = field(default_factory=lambda: [0.0, 0.52, 1.0, 0.88])
    cell_aspect: float = 0.885        # cell height / width (fixed camera perspective)
    cell_shift_up: float = 0.1        # objects are drawn overhanging into the row above
    cell_inset: float = 0.125         # crop inset per side, share of the cell size
    panel_tol: float = 12.0           # LAB distance to the panel background colour
    edge_margin: float = 0.03         # share of width ignored at the frame's left/right edge
    counter_search: float = 0.02      # share of height the k/5 search extends the counter zone
    disc_min_hue_deg: float = 30.0    # face-down tower: centre disc hue vs the rim's hue
    unknown_sand_min: float = 0.03    # share of the picture for an unknown sand colour
    icon_threshold: float = 0.75      # lock / key icon template match (other cells <= 0.62)
    empty_tol: float = 10.0           # LAB distance to the empty-cell colour
    empty_min_frac: float = 0.5       # share of empty-colour pixels for an empty cell
    face_up_max_ratio: float = 0.88   # centre V / top-stage rim V below this = face-up
    face_ref_percentile: float = 85.0  # rim brightness = this percentile of the top part's V
    tower_min_margin: float = 8.0     # LAB margin for a real second colour (tower stage)
    ice_min_white: float = 0.10
    ice_min_dark: float = 0.05
    color_max_dist: float = 22.0      # LAB distance to the nearest palette colour
    sand_max_dist: float = 18.0
    slot_min_cover: float = 0.55      # conveyor: object share of a bucket's run (posts ~0.45)
    bottom_band: float = 0.06         # fallback "exposed" band if the contact layer isn't found
    contact_search: float = 0.02      # share of screen height searched for the picture's bottom edge
    contact_layer: float = 0.008      # contact layer thickness (share of screen height)
    canonical_width: int = 301        # frames are resized to this width for template matching
    template_threshold: float = 0.75
    template_thresholds: Dict[str, float] = field(  # per-anchor overrides
        default_factory=lambda: {"close_red": 0.95, "ad_mute": 0.85})
    template_scales: List[float] = field(
        default_factory=lambda: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3])


@dataclass
class StrategyConfig:
    conveyor_slots: int = 5               # "k/5" on the HUD
    min_exposed: float = 0.03             # share of the picture's bottom band in the colour
    last_slot_min_exposed: float = 0.25   # stricter, when the move takes the last free slot
    sufficiency_margin: float = 0.5       # remaining sand >= margin * need * est. capacity
    unknown_fill_need: float = 0.5        # conveyor bucket without a fill label: still needs
    weight_exposed: float = 1.0           # score per exposed share of the colour
    weight_open_bucket: float = 0.15      # score per bucket the tap connects to the exit
    weight_open_lock: float = 0.05        # score per lock cell the tap connects to the exit
    weight_key_unlock: float = 0.15       # score per bucket under a lock a released key opens
    weight_new_colour: float = 0.05       # bonus for a colour not on the belt yet
    belt_demand_exposure: float = 0.25    # contact-layer share one empty belt bucket claims
    stall_s: float = 4.0                  # belt colour without fill growth this long = stalled
    analyze_interval_s: float = 0.5       # overlay --dry-run: min seconds between analyses
    confirm_frames: int = 2               # consecutive agreeing analyses before a tap
    vision_fail_max_saves: int = 300      # failed reads saved to debug/vision_fail per run


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
    collect: CollectConfig = field(default_factory=CollectConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
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
    if not 0 < cfg.collect.change_threshold <= 1:
        raise ConfigError("collect.change_threshold must be in (0, 1]")
    if cfg.collect.min_interval_s < 0 or cfg.collect.poll_s <= 0:
        raise ConfigError("collect.min_interval_s must be >= 0 and collect.poll_s > 0")
    v = cfg.vision
    for name in ("picture", "conveyor", "fill_text", "counter", "grid_zone"):
        r = getattr(v, name)
        if (not isinstance(r, (list, tuple)) or len(r) != 4
                or not all(isinstance(x, (int, float)) for x in r)
                or not (0 <= r[0] < r[2] <= 1 and 0 <= r[1] < r[3] <= 1)):
            raise ConfigError(f"vision.{name} must be [x0, y0, x1, y1] with 0 <= x0 < x1 <= 1")
    if not v.template_scales or not all(isinstance(s, (int, float)) and s > 0
                                        for s in v.template_scales):
        raise ConfigError("vision.template_scales must be a non-empty list of numbers > 0")
    st = cfg.strategy
    if st.conveyor_slots < 1 or st.confirm_frames < 1:
        raise ConfigError("strategy.conveyor_slots and confirm_frames must be >= 1")
    if st.vision_fail_max_saves < 0:
        raise ConfigError("strategy.vision_fail_max_saves must be >= 0")
    for name in ("min_exposed", "last_slot_min_exposed", "unknown_fill_need"):
        if not 0 <= getattr(st, name) <= 1:
            raise ConfigError(f"strategy.{name} must be in [0, 1]")
    if min(st.sufficiency_margin, st.analyze_interval_s, st.belt_demand_exposure, st.stall_s) < 0:
        raise ConfigError("strategy.sufficiency_margin, analyze_interval_s, "
                          "belt_demand_exposure and stall_s must be >= 0")
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
