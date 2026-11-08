"""Tests for the watching overlay, drawn on a picture already scaled up 3 times."""

from PIL import Image

from kartzero.buttons import Button
from kartzero.overlay import DPAD_BOX, FACE_CENTRE, FACE_GAP, SHOULDERS, SPEED_BOTTOM, SPEED_RIGHT, draw_overlay

SCALE = 3
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def frame():
    return Image.new("RGB", (256 * SCALE, 384 * SCALE), BLACK)


def at(x, y):
    """A DS pixel position on the scaled picture."""
    return round(x * SCALE), round(y * SCALE)


def test_held_buttons_are_filled_in_and_others_are_not():
    drawn = draw_overlay(frame(), Button.A | Button.R | Button.LEFT, 1.0, SCALE)
    middle_y = (DPAD_BOX[1] + DPAD_BOX[3]) / 2
    assert drawn.getpixel(at(DPAD_BOX[0] + 4, middle_y)) == WHITE  # inside the LEFT arm
    assert drawn.getpixel(at(DPAD_BOX[2] - 4, middle_y)) == BLACK  # inside the RIGHT arm
    assert drawn.getpixel(at(FACE_CENTRE[0] + FACE_GAP, FACE_CENTRE[1])) == WHITE  # A
    assert drawn.getpixel(at(FACE_CENTRE[0], FACE_CENTRE[1] + FACE_GAP)) == BLACK  # B
    for button, expected in ((Button.R, WHITE), (Button.L, BLACK)):
        left, top, right, bottom = SHOULDERS[button]
        assert drawn.getpixel(at((left + right) / 2, (top + bottom) / 2)) == expected


def test_buttons_have_a_white_stroke_even_when_not_held():
    drawn = draw_overlay(frame(), Button.NONE, 0.0, SCALE)
    x, y = FACE_CENTRE[0], FACE_CENTRE[1] + FACE_GAP
    column = [drawn.getpixel((round(x * SCALE), row)) for row in range(round((y - 6) * SCALE), round(y * SCALE))]
    assert WHITE in column


def test_the_speed_is_drawn_in_the_bottom_right_of_the_top_screen():
    drawn = draw_overlay(frame(), Button.NONE, 1.0, SCALE)
    corner = drawn.crop((round((SPEED_RIGHT - 60) * SCALE), round((SPEED_BOTTOM - 12) * SCALE), round(SPEED_RIGHT * SCALE), round(SPEED_BOTTOM * SCALE)))
    assert corner.getextrema()[0][1] == 255  # some fully white pixel


def test_the_bottom_screen_and_the_original_are_left_alone():
    original = frame()
    drawn = draw_overlay(original, Button.A | Button.LEFT, 1.0, SCALE)
    assert drawn.crop((0, 192 * SCALE, 256 * SCALE, 384 * SCALE)).getextrema() == ((0, 0), (0, 0), (0, 0))
    assert original.getextrema() == ((0, 0), (0, 0), (0, 0))
