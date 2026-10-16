import pytest

from config import DEFAULT_PATH, ConfigError, config_from_dict, load_config


def test_shipped_config_loads():
    cfg = load_config(DEFAULT_PATH)
    assert cfg.window.owner_name == "iPhone Mirroring"
    assert cfg.window.content_rect is None  # "auto"
    assert cfg.click.backend in ("quartz", "pyautogui")
    assert cfg.safety.max_taps_per_sec > 0


def test_defaults_fill_missing_keys():
    cfg = config_from_dict({"safety": {"dry_run": True}})
    assert cfg.safety.dry_run is True
    assert cfg.safety.max_taps_per_sec == 3.0
    assert cfg.stability.consecutive == 2


def test_content_rect_override():
    cfg = config_from_dict({"window": {"content_rect": [0, 0.04, 1, 0.96]}})
    assert cfg.window.content_rect == (0.0, 0.04, 1.0, 0.96)


@pytest.mark.parametrize("bad", [
    {"window": {"content_rect": [0, 0, 1.2, 1]}},
    {"window": {"content_rect": [0, 0, 1]}},
    {"window": {"owner_nam": "typo"}},
    {"click": {"backend": "xdotool"}},
    {"capture": {"backend": "ffmpeg"}},
    {"safety": {"max_taps_per_sec": 0}},
    {"safety": {"dry_run": "yes"}},
    {"stability": {"threshold": "high"}},
    {"window": "iPhone Mirroring"},
])
def test_invalid_configs_rejected(bad):
    with pytest.raises(ConfigError):
        config_from_dict(bad)


def test_int_promoted_to_float():
    cfg = config_from_dict({"stability": {"threshold": 3}})
    assert isinstance(cfg.stability.threshold, float)


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")
