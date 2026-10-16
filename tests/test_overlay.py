import numpy as np
import pytest

pytest.importorskip("cv2")

from tools.overlay import BAR_H, clear_bar, waiting_frame


def test_clear_bar_blanks_only_the_bottom_strip():
    img = np.full((100, 50, 3), 200, dtype=np.uint8)
    clear_bar(img)
    assert (img[-BAR_H:] == 0).all()
    assert (img[: 100 - BAR_H] == 200).all()


def test_clear_bar_handles_images_shorter_than_the_bar():
    img = np.full((10, 5, 3), 200, dtype=np.uint8)
    clear_bar(img, height=30)
    assert (img == 0).all()


def test_waiting_frame_shape_and_text():
    img = waiting_frame("no on-screen window owned by 'iPhone Mirroring'. " * 5, size=(480, 220))
    assert img.shape == (220, 480, 3) and img.dtype == np.uint8
    assert (img > 128).any()  # some white text was drawn
