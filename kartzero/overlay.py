"""Draws the held buttons and the speed over the top screen, for watching only. Training never sees it.

The left of the top screen shows the L and R shoulder buttons, the D-pad and the four face buttons,
with held ones filled in. The bottom right shows the kart's speed. It's drawn at the DS's own size, so it
gets the same chunky pixels as the game when the window scales up.
"""

from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from kartzero.buttons import Button

OUTLINE = (24, 24, 24)
EDGE = (235, 235, 235)
HELD = (255, 255, 255)
LINE = 2

# The game measures speed in distance per frame (our kart's top speed is 7.72). There's no confirmed
# conversion to km/h, so until there is one, the overlay shows the game's own number.
KMH_PER_SPEED_UNIT: Optional[float] = None

# Positions on the 256 x 192 top screen, below the game's item box in the top-left corner.
SHOULDERS = {Button.L: (6, 54, 38, 60), Button.R: (44, 54, 76, 60)}
DPAD_CENTRE = (21, 85)
ARM = 6  # half the width of each D-pad arm
REACH = 15  # from the centre to the end of each arm
FACE_CENTRE = (60, 85)
FACE_GAP = 11  # from the centre of the diamond to each button
FACE_RADIUS = 5
SPEED_BOTTOM = 186  # the speed sits in the bottom-right corner of the top screen


def draw_overlay(image: Image.Image, held: Button, speed: float) -> Image.Image:
    """A copy of both screens with the held buttons and the speed drawn on the top one."""
    frame = image.convert("RGB")
    draw = ImageDraw.Draw(frame)
    for button, box in SHOULDERS.items():
        _shape(draw, box, button in held, rounded=True)
    _dpad(draw, held)
    _face_buttons(draw, held)
    _speed(draw, frame.width, speed)
    return frame


def _shape(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], held: bool, rounded: bool) -> None:
    fill = HELD if held else None
    if rounded:
        draw.rounded_rectangle(box, radius=3, fill=fill, outline=OUTLINE, width=LINE + 2)
        draw.rounded_rectangle(box, radius=3, fill=fill, outline=EDGE, width=1)
    else:
        draw.ellipse(box, fill=fill, outline=OUTLINE, width=LINE + 2)
        draw.ellipse(box, fill=fill, outline=EDGE, width=1)


def _dpad(draw: ImageDraw.ImageDraw, held: Button) -> None:
    x, y = DPAD_CENTRE
    cross = [
        (x - ARM, y - REACH), (x + ARM, y - REACH), (x + ARM, y - ARM), (x + REACH, y - ARM),
        (x + REACH, y + ARM), (x + ARM, y + ARM), (x + ARM, y + REACH), (x - ARM, y + REACH),
        (x - ARM, y + ARM), (x - REACH, y + ARM), (x - REACH, y - ARM), (x - ARM, y - ARM),
    ]
    draw.polygon(cross, outline=OUTLINE, width=LINE + 2)
    draw.polygon(cross, outline=EDGE, width=1)
    arms = {
        Button.UP: (x - ARM + 2, y - REACH + 2, x + ARM - 2, y - 2),
        Button.DOWN: (x - ARM + 2, y + 2, x + ARM - 2, y + REACH - 2),
        Button.LEFT: (x - REACH + 2, y - ARM + 2, x - 2, y + ARM - 2),
        Button.RIGHT: (x + 2, y - ARM + 2, x + REACH - 2, y + ARM - 2),
    }
    for button, box in arms.items():
        if button in held:
            draw.rectangle(box, fill=HELD)


def _face_buttons(draw: ImageDraw.ImageDraw, held: Button) -> None:
    x, y = FACE_CENTRE
    places = {
        Button.X: (x, y - FACE_GAP),
        Button.A: (x + FACE_GAP, y),
        Button.B: (x, y + FACE_GAP),
        Button.Y: (x - FACE_GAP, y),
    }
    for button, (bx, by) in places.items():
        box = (bx - FACE_RADIUS, by - FACE_RADIUS, bx + FACE_RADIUS, by + FACE_RADIUS)
        _shape(draw, box, button in held, rounded=False)


def _speed(draw: ImageDraw.ImageDraw, width: int, speed: float) -> None:
    if KMH_PER_SPEED_UNIT is None:
        text = f"{speed:.2f} speed"
    else:
        text = f"{round(speed * KMH_PER_SPEED_UNIT)} km/h"
    font = ImageFont.load_default(size=16)
    draw.text((width - 6, SPEED_BOTTOM), text, font=font, fill=HELD, anchor="rd", stroke_width=2, stroke_fill=OUTLINE)
