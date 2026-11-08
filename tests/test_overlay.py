"""Tests for the watching overlay, drawn on a picture already scaled up 3 times."""

from PIL import Image

from kartzero.buttons import Button
from kartzero.overlay import DPAD_BOX, shown_kmh, FACE_CENTRE, FACE_GAP, SHOULDERS, SPEED_BOTTOM, SPEED_RIGHT, draw_overlay

SCALE = 3
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def frame():
    return Image.new("RGB", (256 * SCALE, 384 * SCALE), BLACK)


def at(x, y):
    """A DS pixel position on the scaled picture."""
    return round(x * SCALE), round(y * SCALE)


def test_held_buttons_are_filled_in_and_others_are_not():
    drawn = frame()
    draw_overlay(drawn, Button.A | Button.R | Button.LEFT, 7.72, SCALE)
    middle_y = (DPAD_BOX[1] + DPAD_BOX[3]) / 2
    assert drawn.getpixel(at(DPAD_BOX[0] + 4, middle_y)) == WHITE  # inside the LEFT arm
    assert drawn.getpixel(at(DPAD_BOX[2] - 4, middle_y)) == BLACK  # inside the RIGHT arm
    assert drawn.getpixel(at(FACE_CENTRE[0] + FACE_GAP, FACE_CENTRE[1])) == WHITE  # A
    assert drawn.getpixel(at(FACE_CENTRE[0], FACE_CENTRE[1] + FACE_GAP)) == BLACK  # B
    for button, expected in ((Button.R, WHITE), (Button.L, BLACK)):
        left, top, right, bottom = SHOULDERS[button]
        assert drawn.getpixel(at((left + right) / 2, (top + bottom) / 2)) == expected


def test_buttons_have_a_white_stroke_even_when_not_held():
    drawn = frame()
    draw_overlay(drawn, Button.NONE, 0.0, SCALE)
    x, y = FACE_CENTRE[0], FACE_CENTRE[1] + FACE_GAP
    column = [drawn.getpixel((round(x * SCALE), row)) for row in range(round((y - 6) * SCALE), round(y * SCALE))]
    assert WHITE in column


def test_the_speed_is_drawn_in_the_bottom_right_of_the_top_screen():
    drawn = frame()
    draw_overlay(drawn, Button.NONE, 7.72, SCALE)
    corner = drawn.crop((round((SPEED_RIGHT - 60) * SCALE), round((SPEED_BOTTOM - 12) * SCALE), round(SPEED_RIGHT * SCALE), round(SPEED_BOTTOM * SCALE)))
    assert corner.getextrema()[0][1] == 255  # some fully white pixel


def test_the_bottom_screen_is_left_alone():
    drawn = frame()
    draw_overlay(drawn, Button.A | Button.LEFT, 7.72, SCALE)
    assert drawn.crop((0, 192 * SCALE, 256 * SCALE, 384 * SCALE)).getextrema() == ((0, 0), (0, 0), (0, 0))


def test_sharp_scaling_makes_every_game_pixel_the_same_size():
    from kartzero.window import sharp_scale

    checkerboard = Image.new("L", (8, 8))
    checkerboard.putdata([255 * ((x + y) % 2) for y in range(8) for x in range(8)])
    scaled = sharp_scale(checkerboard, 2.5)
    assert scaled.size == (20, 20)
    row = [scaled.getpixel((x, 10)) for x in range(20)]
    widths, run = [], 1  # the width of each solid run of light or dark pixels, ignoring soft edges
    for left, right in zip(row, row[1:]):
        if (left > 127) == (right > 127):
            run += 1
        else:
            widths.append(run)
            run = 1
    assert max(widths[1:-1]) - min(widths[1:-1]) <= 1


def test_km_h_shows_114_at_boost_speed_and_less_without_a_boost():
    assert shown_kmh(0.0) == 0
    assert shown_kmh(10.04) == 114  # boosting, like the record clips
    assert shown_kmh(7.72) == 88  # full speed without a boost
    assert shown_kmh(5.02) == 57  # half of boost speed
