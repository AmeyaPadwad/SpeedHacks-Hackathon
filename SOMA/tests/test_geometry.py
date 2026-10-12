"""Screen-coordinate mappings stay correct across resizes and window moves."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from soma.perception.preprocess import frame_to_tensor, letterbox_geometry
from soma.perception.screen_parser import (
    iou,
    rescale_pixels,
    screen_to_window,
    to_normalized,
    to_pixels,
    window_to_screen,
)

sizes = st.tuples(st.integers(64, 3840), st.integers(64, 2160))


@st.composite
def norm_boxes(draw):
    x0 = draw(st.floats(0, 0.95))
    y0 = draw(st.floats(0, 0.95))
    x1 = draw(st.floats(x0 + 0.01, 1.0))
    y1 = draw(st.floats(y0 + 0.01, 1.0))
    return (x0, y0, x1, y1)


@settings(max_examples=200, deadline=None)
@given(box=norm_boxes(), size=sizes)
def test_normalized_pixel_round_trip(box, size):
    back = to_normalized(to_pixels(box, size), size)
    w, h = size
    tol = (0.5 / w + 1e-9, 0.5 / h + 1e-9)
    for i, (a, b) in enumerate(zip(box, back)):
        assert abs(a - b) <= tol[i % 2]


@settings(max_examples=200, deadline=None)
@given(box=norm_boxes(), a=sizes, b=sizes)
def test_same_element_across_resizes(box, a, b):
    """A box found at resolution A maps to the same place at resolution B."""
    pa = to_pixels(box, a)
    pb = rescale_pixels(pa, a, b)
    na, nb = to_normalized(pa, a), to_normalized(pb, b)
    for i in range(4):
        dim = 0 if i % 2 == 0 else 1
        assert abs(na[i] - nb[i]) <= 0.5 / a[dim] + 0.5 / b[dim] + 1e-9


@settings(max_examples=200, deadline=None)
@given(box=norm_boxes(), x=st.integers(-2000, 4000), y=st.integers(-1000, 2000),
       w=st.integers(64, 3000), h=st.integers(64, 2000))
def test_window_screen_round_trip(box, x, y, w, h):
    bounds = (x, y, w, h)
    sb = window_to_screen(box, bounds)
    assert x <= sb[0] <= sb[2] <= x + w and y <= sb[1] <= sb[3] <= y + h
    back = screen_to_window(sb, bounds)
    assert all(abs(p - q) <= 0.5 / min(w, h) + 1e-9 for p, q in zip(box, back))


def test_window_move_keeps_normalized_box():
    box = (0.25, 0.5, 0.5, 0.75)
    a = window_to_screen(box, (0, 0, 800, 600))
    b = window_to_screen(box, (300, 200, 800, 600))
    assert (b[0] - a[0], b[1] - a[1]) == (300, 200)
    assert screen_to_window(b, (300, 200, 800, 600)) == pytest.approx(box)


@settings(max_examples=100, deadline=None)
@given(box=norm_boxes(), size=sizes, target=st.sampled_from([64, 224, 448]))
def test_letterbox_round_trip(box, size, target):
    tf = letterbox_geometry(size, target)
    dst = tf.src_to_dst(box)
    assert all(-1e-6 <= v <= 1 + 1e-6 for v in dst)
    back = tf.dst_to_src(dst)
    assert back == pytest.approx(box, abs=1e-6)


def test_letterbox_preserves_aspect_and_content_location():
    img = np.zeros((100, 400, 3), np.uint8)
    img[:, 200:300] = 255                      # white band at x in [0.5, 0.75]
    t, tf = frame_to_tensor(img, 64, normalize=False)
    assert t.shape == (3, 64, 64)
    assert tf.new_size == (64, 16) and tf.pad_y == 24 and tf.pad_x == 0
    x0, y0, x1, y1 = tf.src_to_dst((0.5, 0.0, 0.75, 1.0))
    px0, py0, px1, py1 = to_pixels((x0, y0, x1, y1), (64, 64))
    inside = t[:, py0 + 1:py1 - 1, px0 + 1:px1 - 1]
    assert inside.min() > 0.9
    assert t[:, :tf.pad_y].abs().max() == 0       # letterbox padding is black
    assert t[:, :, : px0 - 1].max() < 0.1


def test_iou_basic():
    assert iou((0, 0, 1, 1), (0, 0, 1, 1)) == pytest.approx(1.0)
    assert iou((0, 0, 0.5, 1), (0.5, 0, 1, 1)) == 0.0
    assert iou((0, 0, 1, 1), (0, 0, 0.5, 1)) == pytest.approx(0.5)
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0
