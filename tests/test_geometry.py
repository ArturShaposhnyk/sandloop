import pytest

from geometry import FULL_WINDOW, Rect, ScreenGeometry, detect_scale, in_corner, is_valid_rel

# A 400x880 pt window at (100, 50) on a Retina display; iPhone screen starts below a 30pt title bar.
WIN = Rect(100, 50, 400, 880)
CONTENT = Rect(0.0, 30 / 880, 1.0, 850 / 880)


@pytest.fixture
def geom() -> ScreenGeometry:
    return ScreenGeometry(WIN, scale=2.0, content_frac=CONTENT)


def test_content_rects(geom):
    c = geom.content_pt
    assert (c.x, c.y, c.w, c.h) == pytest.approx((100, 80, 400, 850))
    p = geom.content_px
    assert (p.x, p.y, p.w, p.h) == pytest.approx((0, 60, 800, 1700))


def test_rel_to_point_corners(geom):
    assert geom.rel_to_point(0, 0) == pytest.approx((100, 80))
    assert geom.rel_to_point(1, 1) == pytest.approx((500, 930))
    assert geom.rel_to_point(0.5, 0.8) == pytest.approx((300, 80 + 0.8 * 850))


def test_rel_to_px(geom):
    assert geom.rel_to_px(0, 0) == pytest.approx((0, 60))
    assert geom.rel_to_px(0.5, 0.5) == pytest.approx((400, 60 + 850))
    assert geom.rel_to_content_px(0.5, 0.5) == pytest.approx((400, 850))


@pytest.mark.parametrize("rx,ry", [(0, 0), (1, 1), (0.25, 0.75), (0.123, 0.987)])
def test_roundtrips(geom, rx, ry):
    assert geom.point_to_rel(*geom.rel_to_point(rx, ry)) == pytest.approx((rx, ry))
    assert geom.px_to_rel(*geom.rel_to_px(rx, ry)) == pytest.approx((rx, ry))
    assert geom.content_px_to_rel(*geom.rel_to_content_px(rx, ry)) == pytest.approx((rx, ry))


def test_pixel_and_point_agree(geom):
    # pixel in window image / scale + window origin == point
    px, py = geom.rel_to_px(0.3, 0.6)
    x, y = geom.rel_to_point(0.3, 0.6)
    assert (WIN.x + px / geom.scale, WIN.y + py / geom.scale) == pytest.approx((x, y))


def test_non_retina_scale():
    g = ScreenGeometry(WIN, scale=1.0, content_frac=FULL_WINDOW)
    assert g.rel_to_px(1, 1) == pytest.approx((400, 880))
    assert g.rel_to_point(1, 1) == pytest.approx((500, 930))


def test_window_move_keeps_relative():
    a = ScreenGeometry(WIN, 2.0, CONTENT)
    b = ScreenGeometry(Rect(700, 300, 400, 880), 2.0, CONTENT)
    assert b.rel_to_point(0.5, 0.5)[0] - a.rel_to_point(0.5, 0.5)[0] == pytest.approx(600)
    assert a.rel_to_px(0.5, 0.5) == b.rel_to_px(0.5, 0.5)


def test_contains_point(geom):
    assert geom.contains_point(*geom.rel_to_point(0.5, 0.5))
    assert not geom.contains_point(300, 60)  # in the title bar
    assert not geom.contains_point(99, 500)


def test_invalid_geometry():
    with pytest.raises(ValueError):
        ScreenGeometry(WIN, scale=0)
    with pytest.raises(ValueError):
        ScreenGeometry(Rect(0, 0, 0, 10), scale=2)


def test_is_valid_rel():
    assert is_valid_rel(0, 1)
    assert not is_valid_rel(-0.01, 0.5)
    assert not is_valid_rel(0.5, 1.01)


@pytest.mark.parametrize("px,pt,expected", [(800, 400, 2.0), (400, 400, 1.0), (801, 400, 2.0), (1200, 400, 3.0)])
def test_detect_scale(px, pt, expected):
    assert detect_scale(px, pt) == expected


def test_in_corner():
    displays = [Rect(0, 0, 1728, 1117), Rect(1728, 0, 1920, 1080)]
    assert in_corner(0, 0, displays, 5)
    assert in_corner(1727, 1116, displays, 5)
    assert in_corner(1728 + 1919, 3, displays, 5)
    assert not in_corner(800, 500, displays, 5)
    assert not in_corner(20, 0, displays, 5)
