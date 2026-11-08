"""The DS buttons. Combine them with |, as in Button.A | Button.LEFT."""

from enum import IntFlag

from desmume.controls import Keys, keymask


class Button(IntFlag):
    """The DS buttons. Combine them with |, as in Button.A | Button.LEFT."""

    NONE = 0
    A = keymask(Keys.KEY_A)
    B = keymask(Keys.KEY_B)
    X = keymask(Keys.KEY_X)
    Y = keymask(Keys.KEY_Y)
    L = keymask(Keys.KEY_L)
    R = keymask(Keys.KEY_R)
    UP = keymask(Keys.KEY_UP)
    DOWN = keymask(Keys.KEY_DOWN)
    LEFT = keymask(Keys.KEY_LEFT)
    RIGHT = keymask(Keys.KEY_RIGHT)
