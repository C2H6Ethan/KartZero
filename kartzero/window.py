"""A resizable window that shows the game, scaled to fit. The emulator's own window can't stretch its picture."""

import math
import tkinter as tk
from typing import Optional

from PIL import Image, ImageTk

from kartzero.buttons import Button
from kartzero.overlay import draw_overlay

TITLE = "KartZero"
DS_WIDTH, DS_HEIGHT = 256, 384  # both screens, one above the other
START_SCALE = 2


def sharp_scale(image: Image.Image, scale: float) -> Image.Image:
    """Scale pixel art up so every game pixel ends up the same size ("sharp bilinear").

    Scaling straight to an uneven size like 2.81 makes some pixels 2 screen pixels wide and others 3,
    which looks jagged. So scale up by the next whole number with hard edges, then smoothly down to the
    exact size, which only softens the pixel edges by a fraction of a pixel.
    """
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    whole = max(1, math.ceil(scale))
    big = image.resize((image.width * whole, image.height * whole), Image.Resampling.NEAREST)
    return big if big.size == size else big.resize(size, Image.Resampling.BOX)


class GameWindow:
    """Shows one image at a time, keeping its shape however the window is resized."""

    def __init__(self) -> None:
        self._closed = False
        self._root = tk.Tk()
        self._root.title(TITLE)
        self._root.configure(background="black")
        self._root.geometry(f"{DS_WIDTH * START_SCALE}x{DS_HEIGHT * START_SCALE}")
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._picture = tk.Label(self._root, background="black", borderwidth=0)
        self._picture.place(relx=0.5, rely=0.5, anchor="center")
        self._photo: ImageTk.PhotoImage | None = None  # tkinter needs this kept alive while it's shown

    @property
    def closed(self) -> bool:
        return self._closed

    def show(self, image: Image.Image, held: Optional[Button] = None, speed: float = 0.0) -> None:
        """Draw the image at the largest size that fits, and let the window respond to the mouse.

        With `held`, the held buttons and the speed are drawn on top, after scaling, so they stay sharp.
        """
        if self._closed:
            return
        self._root.update()  # closing the window happens here, so check again before drawing
        if self._closed:
            return
        scale = min(self._root.winfo_width() / DS_WIDTH, self._root.winfo_height() / DS_HEIGHT)
        picture = sharp_scale(image.convert("RGB"), scale)
        if held is not None:
            draw_overlay(picture, held, speed, scale)
        self._photo = ImageTk.PhotoImage(picture)
        self._picture.configure(image=self._photo)

    def set_label(self, text: str) -> None:
        """Show a short text in the window's title bar, so it never covers the game, or clear it."""
        if self._closed:
            return
        self._root.title(f"{TITLE}  {text}" if text else TITLE)

    def close(self) -> None:
        if not self._closed:
            self._on_close()

    def _on_close(self) -> None:
        self._closed = True
        self._root.destroy()
