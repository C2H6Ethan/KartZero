"""Draws the held buttons and the speed over the top screen, for watching only. Training never sees it.

The style copies the overlay in Mario Kart DS world record clips: hollow shapes with a white stroke
and a fine black edge, filled white while held, in the bottom-left corner of the top screen, and the
speed in a blocky pixel font in the bottom-right corner. Positions and sizes were measured from a
record clip, in DS pixels.

It's drawn after the picture is scaled up, at the window's own resolution, so lines stay thin and
clean instead of becoming chunky blocks. Each look is built once per window size and reused.
"""

from functools import lru_cache
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from kartzero.buttons import Button

TOP_SCREEN = (256, 192)

# Speed is shown so it can be compared with world record clips: 114 km/h at full boost speed. Record
# runs boost almost non-stop, so their number sits near 114. Anything slower shows as its real share
# of that: our kart at full speed without a boost (7.72) shows 88 km/h.
KMH_AT_BOOST_SPEED = 114
BOOST_SPEED = 10.04  # measured speed of Dry Bones in the ROB-BLS during a mini-turbo or a mushroom

# Positions on the 256 x 192 top screen, in DS pixels, measured from a record clip.
SHOULDERS = {Button.L: (5, 151, 31, 157), Button.R: (40, 151, 66, 157)}
DPAD_BOX = (5, 163, 31, 189)
DPAD_ARM = 9  # how wide each arm of the D-pad is
FACE_CENTRE = (52, 176)
FACE_GAP = 9.5  # from the centre of the diamond to each face button
FACE_RADIUS = 5
SPEED_RIGHT = 251  # where the speed text ends
SPEED_BOTTOM = 181  # where the speed text sits

# Line widths, in DS pixels: a fine black edge, a white stroke, then another fine black edge.
EDGE = 0.4
STROKE = 0.8

BLACK = (0, 0, 0, 255)
WHITE = (255, 255, 255, 255)
CLEAR = (0, 0, 0, 0)

Box = tuple[float, float, float, float]


def draw_overlay(picture: Image.Image, held: Button, speed: float, scale: float) -> None:
    """Draw the held buttons and the speed onto the picture, which is already scaled up by `scale`.

    It draws in place, pasting ready-made layers, because this runs for every frame the window shows.
    """
    buttons = _buttons(round(scale, 2), int(held))
    picture.paste(buttons, (0, 0), buttons)
    words = _text(f"{shown_kmh(speed)} km/h", max(1, round(scale)))
    picture.paste(words, (round(SPEED_RIGHT * scale) - words.width, round(SPEED_BOTTOM * scale) - words.height), words)


def shown_kmh(speed: float) -> int:
    """The km/h to show for a speed in the game's units: its share of boost speed, which shows 114."""
    return round(KMH_AT_BOOST_SPEED * speed / BOOST_SPEED)


@lru_cache(maxsize=64)
def _buttons(scale: float, held: int) -> Image.Image:
    """Every button outline, and the held ones filled in, on a clear layer the size of the scaled top screen."""
    outlines, fills = _shapes(scale)
    layer = Image.new("RGBA", _scaled_size(scale), CLEAR)
    edge, stroke = _line_widths(scale)
    for shape in outlines:
        inside_edge = _shrink(shape, edge)
        inside_stroke = _shrink(inside_edge, stroke)
        layer.paste(BLACK, mask=ImageChops.subtract(shape, inside_edge))
        layer.paste(WHITE, mask=ImageChops.subtract(inside_edge, inside_stroke))
        layer.paste(BLACK, mask=ImageChops.subtract(inside_stroke, _shrink(inside_stroke, edge)))
    for button, inside in fills.items():
        if held & int(button):
            layer.paste(WHITE, mask=inside)
    return layer


@lru_cache(maxsize=8)
def _shapes(scale: float) -> tuple[list[Image.Image], dict[Button, Image.Image]]:
    """The shapes to outline, and the inside of each button, filled while it's held, at this scale."""
    edge, stroke = _line_widths(scale)
    lines = 2 * edge + stroke

    def solid(box: Box, round_ends: bool = False, circle: bool = False) -> Image.Image:
        corners = (round(box[0] * scale), round(box[1] * scale), round(box[2] * scale), round(box[3] * scale))
        mask = Image.new("L", _scaled_size(scale), 0)
        draw = ImageDraw.Draw(mask)
        if circle:
            draw.ellipse(corners, fill=255)
        elif round_ends:
            draw.rounded_rectangle(corners, radius=(corners[3] - corners[1]) // 2, fill=255)
        else:
            draw.rectangle(corners, fill=255)
        return mask

    outlines: list[Image.Image] = []
    fills: dict[Button, Image.Image] = {}
    for button, shoulder in SHOULDERS.items():
        shape = solid(shoulder, round_ends=True)
        outlines.append(shape)
        fills[button] = _shrink(shape, lines)

    left, top, right, bottom = DPAD_BOX
    middle_x, middle_y, half = (left + right) / 2, (top + bottom) / 2, DPAD_ARM / 2
    # Each arm stops at the centre square, so a held direction fills only its own arm.
    arms: dict[Button, Box] = {
        Button.UP: (middle_x - half, top, middle_x + half, middle_y - half),
        Button.DOWN: (middle_x - half, middle_y + half, middle_x + half, bottom),
        Button.LEFT: (left, middle_y - half, middle_x - half, middle_y + half),
        Button.RIGHT: (middle_x + half, middle_y - half, right, middle_y + half),
    }
    cross = solid((middle_x - half, middle_y - half, middle_x + half, middle_y + half))
    for box in arms.values():
        cross = ImageChops.lighter(cross, solid(box))
    outlines.append(cross)  # the D-pad is outlined once, as a whole cross
    inside_cross = _shrink(cross, lines)
    for button, box in arms.items():
        fills[button] = ImageChops.multiply(inside_cross, solid(box))

    x, y = FACE_CENTRE
    faces = {Button.X: (x, y - FACE_GAP), Button.A: (x + FACE_GAP, y), Button.B: (x, y + FACE_GAP), Button.Y: (x - FACE_GAP, y)}
    for button, (bx, by) in faces.items():
        shape = solid((bx - FACE_RADIUS, by - FACE_RADIUS, bx + FACE_RADIUS, by + FACE_RADIUS), circle=True)
        outlines.append(shape)
        fills[button] = _shrink(shape, lines)
    return outlines, fills


def _scaled_size(scale: float) -> tuple[int, int]:
    return round(TOP_SCREEN[0] * scale), round(TOP_SCREEN[1] * scale)


def _line_widths(scale: float) -> tuple[int, int]:
    """The black edge and white stroke widths in screen pixels, at least 1 each."""
    return max(1, round(EDGE * scale)), max(1, round(STROKE * scale))


def _shrink(mask: Image.Image, pixels: int) -> Image.Image:
    """The shape with `pixels` taken off every edge."""
    return mask.filter(ImageFilter.MinFilter(pixels * 2 + 1)) if pixels > 0 else mask


@lru_cache(maxsize=256)
def _text(text: str, pixel: int) -> Image.Image:
    """The text in the blocky pixel font, `pixel` screen pixels per font pixel, white with a black edge."""
    glyphs = [GLYPHS[character] for character in text]
    width = sum(len(glyph[0]) + 1 for glyph in glyphs) - 1
    border = max(1, pixel // 2)
    shape = Image.new("L", (width * pixel + 2 * border, GLYPH_HEIGHT * pixel + 2 * border), 0)
    draw = ImageDraw.Draw(shape)
    x = 0
    for glyph in glyphs:
        for row, line in enumerate(glyph):
            for column, mark in enumerate(line):
                if mark == "#":
                    left, top = border + (x + column) * pixel, border + row * pixel
                    draw.rectangle((left, top, left + pixel - 1, top + pixel - 1), fill=255)
        x += len(glyph[0]) + 1
    words = Image.new("RGBA", shape.size, CLEAR)
    words.paste(BLACK, mask=shape.filter(ImageFilter.MaxFilter(border * 2 + 1)))
    words.paste(WHITE, mask=shape)
    return words


# A blocky font with 2-pixel-wide strokes, like the record clips' speedometer. Each row is one line of pixels.
GLYPHS = {
    "0": [".####.", "##..##", "##..##", "##..##", "##..##", "##..##", "##..##", ".####."],
    "1": ["..##..", ".###..", "..##..", "..##..", "..##..", "..##..", "..##..", ".####."],
    "2": [".####.", "##..##", "....##", "...##.", "..##..", ".##...", "##....", "######"],
    "3": [".####.", "##..##", "....##", "..###.", "....##", "....##", "##..##", ".####."],
    "4": ["...##.", "..###.", ".####.", "##.##.", "##.##.", "######", "...##.", "...##."],
    "5": ["######", "##....", "##....", "#####.", "....##", "....##", "##..##", ".####."],
    "6": [".####.", "##....", "##....", "#####.", "##..##", "##..##", "##..##", ".####."],
    "7": ["######", "....##", "...##.", "...##.", "..##..", "..##..", ".##...", ".##..."],
    "8": [".####.", "##..##", "##..##", ".####.", "##..##", "##..##", "##..##", ".####."],
    "9": [".####.", "##..##", "##..##", "##..##", ".#####", "....##", "....##", ".####."],
    ".": ["..", "..", "..", "..", "..", "..", "##", "##"],
    " ": ["..", "..", "..", "..", "..", "..", "..", ".."],
    "k": ["##....", "##....", "##..##", "##.##.", "####..", "##.##.", "##..##", "##..##"],
    "m": ["........", "........", "#######.", "##.##.##", "##.##.##", "##.##.##", "##.##.##", "##.##.##"],
    "/": ["....##", "....##", "...##.", "...##.", "..##..", ".##...", ".##...", "##...."],
    "h": ["##....", "##....", "#####.", "##..##", "##..##", "##..##", "##..##", "##..##"],
}
GLYPH_HEIGHT = 8
