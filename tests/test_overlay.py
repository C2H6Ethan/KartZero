"""Tests for the watching overlay."""

from PIL import Image

from kartzero.buttons import Button
from kartzero.overlay import DPAD_CENTRE, FACE_CENTRE, FACE_GAP, HELD, SHOULDERS, draw_overlay

BLACK = (0, 0, 0)


def frame():
    return Image.new("RGB", (256, 384), BLACK)


def middle(box):
    return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)


def test_held_buttons_are_filled_in_and_others_are_not():
    drawn = draw_overlay(frame(), Button.A | Button.R | Button.LEFT, 7.0)
    left_arm = (DPAD_CENTRE[0] - 8, DPAD_CENTRE[1])
    right_arm = (DPAD_CENTRE[0] + 8, DPAD_CENTRE[1])
    a_button = (FACE_CENTRE[0] + FACE_GAP, FACE_CENTRE[1])
    b_button = (FACE_CENTRE[0], FACE_CENTRE[1] + FACE_GAP)
    assert drawn.getpixel(left_arm) == HELD
    assert drawn.getpixel(right_arm) != HELD
    assert drawn.getpixel(a_button) == HELD
    assert drawn.getpixel(b_button) != HELD
    assert drawn.getpixel(middle(SHOULDERS[Button.R])) == HELD
    assert drawn.getpixel(middle(SHOULDERS[Button.L])) != HELD


def test_the_bottom_screen_is_left_alone():
    drawn = draw_overlay(frame(), Button.A, 7.0)
    assert all(drawn.getpixel((x, y)) == BLACK for x in range(0, 256, 8) for y in range(192, 384, 8))


def test_the_original_frame_is_not_changed():
    original = frame()
    draw_overlay(original, Button.A, 7.0)
    assert original.getpixel((FACE_CENTRE[0] + FACE_GAP, FACE_CENTRE[1])) == BLACK
