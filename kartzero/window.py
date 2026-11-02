"""A resizable window that shows the game, scaled to fit. The emulator's own window can't stretch its picture."""

import tkinter as tk

from PIL import Image, ImageTk

TITLE = "KartZero"
DS_WIDTH, DS_HEIGHT = 256, 384  # both screens, one above the other
START_SCALE = 2


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

    def show(self, image: Image.Image) -> None:
        """Draw the image at the largest size that fits, and let the window respond to the mouse."""
        if self._closed:
            return
        self._root.update()  # closing the window happens here, so check again before drawing
        if self._closed:
            return
        scale = min(self._root.winfo_width() / DS_WIDTH, self._root.winfo_height() / DS_HEIGHT)
        size = (max(1, round(DS_WIDTH * scale)), max(1, round(DS_HEIGHT * scale)))
        self._photo = ImageTk.PhotoImage(image.resize(size, Image.Resampling.NEAREST))
        self._picture.configure(image=self._photo)

    def close(self) -> None:
        if not self._closed:
            self._on_close()

    def _on_close(self) -> None:
        self._closed = True
        self._root.destroy()
