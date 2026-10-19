"""Typed configuration loaded from config.yaml (pure Python, testable anywhere)."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

DEFAULT_PATH = Path(__file__).resolve().parent / "config.yaml"


VISION_ZONES = ("picture", "conveyor", "fill_text", "counter", "grid_zone")


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
    aspect: float = 0.4624             # iPhone screen width / height (602 x 1302 px)
    aspect_tolerance: float = 0.02     # relative; outside = window partly off-screen / moving


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
    unlock_trust_s: float = 10.0      # after a key lock opens, its cells follow the path rule alone
    cell_inset: float = 0.125         # crop inset per side, share of the cell size
    panel_tol: float = 12.0           # LAB distance to the panel background colour
    edge_margin: float = 0.03         # share of width ignored at the frame's left/right edge
    counter_search: float = 0.02      # share of height the k/5 search extends the counter zone
    disc_min_dist: float = 20.0       # face-down tower: LAB distance disc vs the ring around it
    unknown_sand_min: float = 0.03    # share of the picture for an unknown sand colour
    icon_threshold: float = 0.75      # lock / key icon template match (other cells <= 0.62)
    chest_tol: float = 12.0           # LAB distance to palette ui "chest_wood"
    chest_min_wood: float = 0.2       # chest: wood share (L186 chests 0.33, others <= 0.06)
    mystery_tol: float = 10.0         # LAB distance to palette ui "mystery_grey"
    mystery_min_grey: float = 0.3     # mystery bucket: share of grey pixels (L186: 0.5)
    lattice_switch_frames: int = 3    # a different grid geometry must hold this many frames
    hint_threshold: float = 0.8       # hint arrow template match (hints.py)
    hint_scales: List[float] = field(default_factory=lambda: [0.9, 1.0, 1.1])
    hint_downscale: float = 0.5       # hint arrows are matched on the grid zone scaled by this
    hint_cover_max: float = 0.5       # cell crop share under an arrow above which it's occluded
    empty_tol: float = 10.0           # LAB distance to the empty-cell colour
    empty_min_frac: float = 0.5       # share of empty-colour pixels for an empty cell
    face_up_max_ratio: float = 0.88   # centre V / top-stage rim V below this = face-up
    colour_watch: List[List[str]] = field(  # close bucket colour pairs: always in colour_notes
        default_factory=lambda: [["green", "darkgreen"]])
    rim_percentile: float = 75.0      # open bucket colour = object pixels of the top stage this
                                      # bright and up (the rim; the inside is shaded)
    face_ref_percentile: float = 85.0  # rim brightness = this percentile of the top part's V
    tower_min_margin: float = 8.0     # LAB margin for a real second colour (tower stage)
    ice_min_white: float = 0.10
    ice_min_dark: float = 0.05
    color_max_dist: float = 22.0      # LAB distance to the nearest palette colour
    sand_max_dist: float = 18.0
    slot_min_cover: float = 0.55      # conveyor: object share of a bucket's run (posts ~0.45)
    bottom_band: float = 0.06         # fallback "exposed" band if the contact layer isn't found
    lower_band: float = 0.3           # "lower part" of the picture zone (sand reaching soon)
    contact_search: float = 0.02      # share of screen height searched for the picture's bottom edge
    contact_layer: float = 0.022      # sand a bucket reaches: lowest px of sand per column (share
                                      # of screen height, ~28 px at 1302)
    contact_reach: float = 0.07       # ... searched this far above the bottom edge
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
    weight_open_bucket: float = 0.05      # score per bucket the tap connects (any colour)
    weight_open_lock: float = 0.05        # score per lock cell the tap connects to the exit
    weight_key_unlock: float = 0.15       # score per bucket under a lock a released key opens
    weight_tower_short_stage: float = 0.5  # penalty per later tower stage short of sand
    weight_new_colour: float = 0.05       # bonus for a colour not on the belt yet
    # unlock value / path planning (planner.py)
    lower_weight: float = 0.5             # usefulness counts the lower-part share x this
    useful_min: float = 0.05              # usefulness from which a colour counts as useful
    weight_unlock_value: float = 1.0      # score per usefulness of every useful opened bucket
    path_key_value: float = 0.3           # target value of a key, per bucket under its lock
    path_good_avail: float = 0.3          # path step costs 1 + extra below this avail
    path_cost_low_exposure: float = 1.0   # ... up to this extra (avail 0)
    path_step_cost: float = 0.02          # plan rank = target value - this x path cost
    weight_path_bonus: float = 1.0        # bonus for the first step = this x plan rank
    belt_demand_exposure: float = 0.25    # contact-layer share one empty belt bucket claims
    relaxed_free_slots: int = 3           # this many free slots and up: min exposure relaxed
    relaxed_min_exposed: float = 0.0      # ... to this (also when no belt bucket collects)
    opener_free_slots: int = 2            # a tap that opens a useful bucket / a path step,
    opener_min_exposed: float = 0.0       # ... with this many free slots: this minimum
    stall_s: float = 4.0                  # belt colour without fill growth this long = stalled
    stall_min_growth: int = 2             # fill points that count as growth (labels jitter by 1)
    stall_forget_s: float = 3.0           # a belt colour out of sight this long is forgotten
    weight_stalled_colour: float = 2.0    # penalty: one more bucket of a stalled colour = lost slot
    path_cost_stalled: float = 3.0        # extra path cost of a step whose colour is stalled
    # L186 chests / mystery buckets (bot heuristics on the user's rules, GAME_RULES.md)
    weight_chest_cut: float = 0.3         # penalty per expected path cell a chest scatter fills
    weight_chest_free: float = 0.3        # penalty x share of free cells (prefer few free cells)
    weight_mystery_reveal: float = 0.05   # bonus per mystery bucket the tap reveals
    weight_mystery_reachable: float = 0.15  # ... extra if it is connected to the exit after
    chest_settle_s: float = 1.5           # after a chest opened: no decision this long
    analyze_interval_s: float = 0.5       # overlay --dry-run: min seconds between analyses
    confirm_frames: int = 2               # consecutive agreeing analyses before a tap
    vision_fail_max_saves: int = 300      # failed reads saved to debug/vision_fail per run


@dataclass
class BotConfig:
    """bot.py: the playing loop (tempo, settle after taps, stop conditions, run logs)."""
    settle_min_s: float = 0.25           # after a tap: wait at least this long
    settle_max_s: float = 1.5            # ... and at most this long for the grid to calm down
    settle_threshold: float = 0.01       # frame change (0..1) in settle_zones below = settled
    settle_zones: List[str] = field(default_factory=lambda: ["grid_zone", "counter"])
    burst_max: int = 3                   # taps released back to back from one confirmed decision
    burst_min_free: int = 2              # a follow-up tap needs this many free slots (never the last)
    burst_min_avail: float = 0.05        # ... and its colour this much avail (useful right now)
    burst_gap_s: float = 0.3             # min seconds between burst taps (rate limit applies too)
    cover_window_s: float = 2.0          # exposure trend window for "about to be covered"
    cover_min_drop: float = 0.03         # contact-layer share drop in that window that counts
    weight_covering: float = 1.0         # score bonus per dropped share (colour not on the belt)
    retap_guard_s: float = 2.0           # never tap a cell we tapped this recently again
    max_no_effect_taps: int = 2          # taps in a row that changed nothing -> stop
    screen_confirm_frames: int = 3       # non-gameplay this many analyses in a row -> stop
    win_sand_max: float = 0.01           # level ends with less sand than this -> inferred win
    window_lost_max_s: float = 10.0      # window missing / geometry check failing this long -> stop
    max_run_s: float = 900.0             # safety stop for one level
    disagree_report_s: float = 10.0      # look != path rule this long with no move: log + frame
    loop_interval_s: float = 0.05        # min seconds between captures
    max_saved_frames: int = 1500         # PNGs per run in debug/runs/<stamp>/frames/


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
    bot: BotConfig = field(default_factory=BotConfig)
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
    if cfg.screen.aspect <= 0 or cfg.screen.aspect_tolerance <= 0:
        raise ConfigError("screen.aspect and screen.aspect_tolerance must be > 0")
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
    for name in VISION_ZONES:
        r = getattr(v, name)
        if (not isinstance(r, (list, tuple)) or len(r) != 4
                or not all(isinstance(x, (int, float)) for x in r)
                or not (0 <= r[0] < r[2] <= 1 and 0 <= r[1] < r[3] <= 1)):
            raise ConfigError(f"vision.{name} must be [x0, y0, x1, y1] with 0 <= x0 < x1 <= 1")
    if not v.template_scales or not all(isinstance(s, (int, float)) and s > 0
                                        for s in v.template_scales):
        raise ConfigError("vision.template_scales must be a non-empty list of numbers > 0")
    if min(v.chest_tol, v.chest_min_wood, v.mystery_tol, v.mystery_min_grey,
           v.unlock_trust_s) < 0 or v.lattice_switch_frames < 1:
        raise ConfigError("vision chest / mystery / unlock settings must be >= 0 and "
                          "lattice_switch_frames >= 1")
    st = cfg.strategy
    if st.conveyor_slots < 1 or st.confirm_frames < 1:
        raise ConfigError("strategy.conveyor_slots and confirm_frames must be >= 1")
    if st.vision_fail_max_saves < 0:
        raise ConfigError("strategy.vision_fail_max_saves must be >= 0")
    for name in ("min_exposed", "last_slot_min_exposed", "unknown_fill_need"):
        if not 0 <= getattr(st, name) <= 1:
            raise ConfigError(f"strategy.{name} must be in [0, 1]")
    if min(st.lower_weight, st.useful_min, st.weight_unlock_value, st.path_key_value,
           st.path_good_avail, st.path_cost_low_exposure, st.path_step_cost,
           st.weight_path_bonus) < 0:
        raise ConfigError("strategy: unlock value / path weights must be >= 0")
    if min(st.sufficiency_margin, st.analyze_interval_s, st.belt_demand_exposure, st.stall_s,
           st.stall_min_growth, st.stall_forget_s, st.weight_stalled_colour,
           st.path_cost_stalled, st.weight_chest_cut, st.weight_chest_free,
           st.weight_mystery_reveal, st.weight_mystery_reachable, st.chest_settle_s) < 0:
        raise ConfigError("strategy.sufficiency_margin, analyze_interval_s, "
                          "belt_demand_exposure and the stall settings must be >= 0")
    b = cfg.bot
    if b.burst_max < 1 or b.burst_min_free < 2 or b.screen_confirm_frames < 1:
        raise ConfigError("bot.burst_max and screen_confirm_frames must be >= 1, "
                          "burst_min_free >= 2 (a burst never takes the last slot)")
    if b.disagree_report_s <= 0:
        raise ConfigError("bot.disagree_report_s must be > 0")
    if b.max_no_effect_taps < 1 or b.max_saved_frames < 0:
        raise ConfigError("bot.max_no_effect_taps must be >= 1, max_saved_frames >= 0")
    if min(b.settle_min_s, b.settle_threshold, b.burst_min_avail, b.burst_gap_s,
           b.cover_window_s, b.cover_min_drop, b.weight_covering, b.retap_guard_s,
           b.win_sand_max, b.loop_interval_s) < 0 or b.settle_max_s < b.settle_min_s:
        raise ConfigError("bot: times / thresholds must be >= 0, settle_max_s >= settle_min_s")
    if b.window_lost_max_s <= 0 or b.max_run_s <= 0:
        raise ConfigError("bot.window_lost_max_s and max_run_s must be > 0")
    for z in b.settle_zones:
        if z not in VISION_ZONES:
            raise ConfigError(f"bot.settle_zones: unknown vision zone {z!r}")
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
