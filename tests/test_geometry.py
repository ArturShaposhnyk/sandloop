import numpy as np
import pytest

from actions import OutOfBounds, target_point
from capture import (Rect, ScreenGeometry, WindowInfo, WindowNotFound, content_from_alpha,
                     content_from_relative, crop_points,
                     content_from_window_pixels, detect_content_rect, pick_window)


def geo(scale: float = 2.0) -> ScreenGeometry:
    window = Rect(100, 50, 400, 900)
    content = Rect(110, 90, 380, 820)  # 10pt side frame, 40pt title bar, 40pt bottom
    return ScreenGeometry(window, content, scale)


def test_rel_to_point_corners_and_center():
    g = geo()
    assert g.rel_to_point(0, 0) == (110, 90)
    assert g.rel_to_point(1, 1) == (490, 910)
    assert g.rel_to_point(0.5, 0.5) == (300, 500)


@pytest.mark.parametrize("rx,ry", [(0.0, 0.0), (0.25, 0.8), (0.999, 0.001), (1.0, 1.0)])
def test_point_rel_roundtrip(rx, ry):
    g = geo()
    assert g.point_to_rel(*g.rel_to_point(rx, ry)) == pytest.approx((rx, ry))


@pytest.mark.parametrize("scale", [1.0, 2.0, 3.0, 1.5])
def test_pixel_conversion_scale(scale):
    g = geo(scale)
    assert g.frame_size == (round(380 * scale), round(820 * scale))
    px, py = g.rel_to_pixel(0.5, 0.25)
    assert (px, py) == (int(0.5 * 380 * scale), int(0.25 * 820 * scale))
    assert g.pixel_to_rel(px, py) == pytest.approx((0.5, 0.25), abs=1 / (380 * scale))
    # far edge clamps to the last valid pixel index
    assert g.rel_to_pixel(1, 1) == (g.frame_size[0] - 1, g.frame_size[1] - 1)


@pytest.mark.parametrize("rx,ry", [(-0.01, 0.5), (0.5, 1.01), (2, 2)])
def test_out_of_range_rejected(rx, ry):
    with pytest.raises(ValueError):
        geo().rel_to_point(rx, ry)
    with pytest.raises(OutOfBounds):
        target_point(geo(), rx, ry)


def test_window_pixels_to_points_retina():
    window = Rect(100, 50, 400, 900)
    content = content_from_window_pixels(window, Rect(20, 80, 760, 1640), scale=2.0)
    assert content == Rect(110, 90, 380, 820)


def test_relative_override():
    window = Rect(100, 50, 400, 900)
    c = content_from_relative(window, [0.025, 0.1, 0.95, 0.8])
    assert (c.x, c.y, c.w, c.h) == pytest.approx((110, 140, 380, 720))


def test_detect_content_trims_title_bar_and_frame():
    img = np.full((900, 400, 3), 30, np.uint8)               # uniform dark chrome
    rng = np.random.default_rng(0)
    img[40:860, 10:390] = rng.integers(0, 255, (820, 380, 3), dtype=np.uint8)  # "game"
    r = detect_content_rect(img, max_trim=0.15, tolerance=6.0)
    assert (r.x, r.y, r.w, r.h) == (10, 40, 380, 820)


def test_detect_content_respects_max_trim():
    img = np.full((100, 100, 3), 30, np.uint8)
    r = detect_content_rect(img, max_trim=0.1)
    assert (r.x, r.y, r.w, r.h) == (10, 10, 80, 80)


def test_detect_content_keeps_full_window_without_chrome():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (200, 100, 3), dtype=np.uint8)
    r = detect_content_rect(img)
    assert (r.x, r.y, r.w, r.h) == (0, 0, 100, 200)


def _win(owner, w, h, layer=0, title="", wid=1):
    return WindowInfo(wid, owner, title, 42, layer, Rect(0, 0, w, h))


def test_pick_window_largest_matching_owner_case_insensitive():
    wins = [_win("Finder", 800, 600), _win("iPhone Mirroring", 60, 60, wid=2),
            _win("iphone mirroring", 400, 900, wid=3), _win("iPhone Mirroring", 500, 900, layer=25)]
    assert pick_window(wins, "iPhone Mirroring").window_id == 3


def test_pick_window_title_filter_and_not_found():
    wins = [_win("iPhone Mirroring", 400, 900, title="iPhone", wid=7)]
    assert pick_window(wins, "iPhone Mirroring", "iphone").window_id == 7
    with pytest.raises(WindowNotFound):
        pick_window(wins, "iPhone Mirroring", "ipad")
    with pytest.raises(WindowNotFound):
        pick_window(wins, "Повтор iPhone")


def test_content_from_alpha_matches_real_mirroring_layout():
    # Measured on iPhone Mirroring @2x: window 634x1394 px, opaque screen (16,76)-(617,1377).
    alpha = np.zeros((1394, 634), np.uint8)
    alpha[76:1378, 16:618] = 255
    alpha[76:90, 16:40] = 0      # rounded corner cut-outs don't shrink the box
    alpha[60:76, 16:618] = 120   # semi-transparent shadow is ignored
    r = content_from_alpha(alpha)
    assert (r.x, r.y, r.w, r.h) == (16, 76, 602, 1302)
    window = Rect(455, 64, 317, 697)
    assert content_from_window_pixels(window, r, 2.0) == Rect(463, 102, 301, 651)
    assert content_from_alpha(np.zeros((10, 10), np.uint8)) is None


def test_crop_points_retina():
    img = np.arange(40 * 20).reshape(40, 20)
    out = crop_points(img, Rect(100, 100, 10, 20), Rect(102, 105, 5, 10), 2.0)
    assert out.shape == (20, 10)
    assert out[0, 0] == img[10, 4]
