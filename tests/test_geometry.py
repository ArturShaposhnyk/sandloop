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


def _mirroring_window(chrome: bool):
    """634x1394 px window like iPhone Mirroring: screen at (16, 76) 602x1302 px."""
    h, w = 1394, 634
    img = np.zeros((h, w, 3), np.uint8)
    alpha = np.zeros((h, w), np.uint8)
    img[76:1378, 16:618] = (40, 160, 220)  # game screen
    img[76:90, 300:340] = 0                # dynamic island (black, inside the screen)
    alpha[76:1378, 16:618] = 255
    if chrome:
        alpha[:] = 255
        frame = np.ones((h, w), bool)
        frame[76:1378, 16:618] = False
        img[frame] = (45, 45, 46)
        for cx in (32, 78, 124):           # traffic lights in the title bar
            img[26:50, cx - 12:cx + 12] = (90, 90, 255)
        img[26:52, 540:600] = (200, 200, 200)  # toolbar icons
        alpha[:20, :20] = alpha[:20, -20:] = alpha[-20:, :20] = alpha[-20:, -20:] = 0  # corners
    return img, alpha


def test_content_inside_chrome_finds_screen_under_title_bar():
    from capture import content_inside_chrome

    img, alpha = _mirroring_window(chrome=True)
    assert content_from_alpha(alpha) == Rect(0, 0, 634, 1394)  # alpha alone sees the whole window
    assert content_inside_chrome(img, alpha) == Rect(16, 76, 602, 1302)


def test_chrome_and_clean_states_give_the_same_screen_rect():
    from capture import content_inside_chrome

    clean_img, clean_alpha = _mirroring_window(chrome=False)
    img, alpha = _mirroring_window(chrome=True)
    assert content_from_alpha(clean_alpha) == content_inside_chrome(img, alpha)


def test_content_inside_chrome_none_without_chrome():
    from capture import content_inside_chrome

    img = np.full((200, 100, 3), (40, 160, 220), np.uint8)
    assert content_inside_chrome(img, np.full((200, 100), 255, np.uint8)) is None


class _FakeGrabber:
    def __init__(self, img, alpha):
        self.img, self.alpha = img, alpha

    def grab_window(self, win):
        return self.img, self.alpha


def _with_soft_edge(img, alpha, px: int = 2):
    """Window's anti-aliased outer edge: semi-transparent, near-black (seen in real captures)."""
    img, alpha = img.copy(), alpha.copy()
    for a in (img, alpha):
        a[:px] = a[-px:] = 0
        a[:, :px] = a[:, -px:] = 0
    alpha[:px] = alpha[-px:] = 120
    alpha[:, :px] = alpha[:, -px:] = 120
    return img, alpha


@pytest.mark.parametrize("chrome", [False, True])
@pytest.mark.parametrize("soft_edge", [False, True])
def test_resolve_geometry_same_screen_rect_in_all_window_states(chrome, soft_edge):
    from capture import resolve_geometry
    from config import Config

    img, alpha = _mirroring_window(chrome)
    if soft_edge:
        img, alpha = _with_soft_edge(img, alpha)
    win = WindowInfo(1, "iPhone Mirroring", "iPhone Mirroring", 42, 0, Rect(251, 252, 317, 697))
    _, g = resolve_geometry(Config(), _FakeGrabber(img, alpha), windows=[win])
    assert g.content == Rect(259, 290, 301, 651)  # (8, 38) pt inside the window
