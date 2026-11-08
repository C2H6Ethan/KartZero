"""Runs Mario Kart DS in the emulator, with no window and simple controls."""

import ctypes
import sys
import time
from pathlib import Path
from types import TracebackType
from typing import Optional, Self, Union

from desmume.emulator import DeSmuME

from kartzero.buttons import Button
from kartzero.kart import Kart
from kartzero.track import Track
from kartzero.window import GameWindow

ROOT = Path(__file__).resolve().parent.parent
ROM = ROOT / "game" / "mkds.nds"
SAVE = ROOT / "game" / "mkds.sav"
SNAPSHOT = ROOT / "savestates" / "f8c_time_trial.dst"
WATCH_FRAME_SECONDS = 1 / 60
USE_JIT_OFFSET = 0xF54FFC5  # in libdesmume.dll from py-desmume-mkds 0.0.20
JIT_BLOCK_SIZE_DISTANCE = 3
JIT_DEFAULT_BLOCK_SIZE = 12

FilePath = Union[str, Path]


class Game:
    """One running copy of the game.

    Only one can run per process, because a second emulator in the same process crashes.
    """

    def __init__(self, rom: FilePath = ROM) -> None:
        self._emulator = DeSmuME()
        self._enable_jit()
        self._emulator.open(str(rom))
        self._emulator.volume_set(0)
        self.kart = Kart(self._emulator.memory)
        self.track = Track(self._emulator.memory)
        self._window: Optional[GameWindow] = None
        self._next_frame_time = 0.0
        self._held = Button.NONE  # shown in the window's overlay

    def _enable_jit(self) -> None:
        """Run the game's code as compiled code, about 1.8 times faster with identical results.

        The library has no setting for this, so set the byte the emulator reads. Its place
        is specific to this exact DLL, so check it looks right before writing to it.
        """
        address = self._emulator.lib._handle + USE_JIT_OFFSET
        use_jit = ctypes.c_ubyte.from_address(address)
        block_size = ctypes.c_uint32.from_address(address + JIT_BLOCK_SIZE_DISTANCE)
        if (use_jit.value, block_size.value) != (0, JIT_DEFAULT_BLOCK_SIZE):
            raise RuntimeError("the emulator's JIT setting isn't where expected: did py-desmume-mkds change version?")
        use_jit.value = 1

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: Optional[type[BaseException]],
        exception: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._window:
            self._window.close()
        self._emulator.destroy()

    def load_save(self, save: FilePath = SAVE) -> None:
        """Load a save file and restart the game from the title screen."""
        if not self._emulator.backup.import_file(str(save)):
            raise RuntimeError(f"could not load save file {save}")
        self._emulator.reset()

    def load_snapshot(self, snapshot: FilePath = SNAPSHOT) -> None:
        self._emulator.savestate.load_file(str(snapshot))

    def save_snapshot(self, snapshot: FilePath = SNAPSHOT) -> None:
        Path(snapshot).parent.mkdir(exist_ok=True)
        self._emulator.savestate.save_file(str(snapshot))

    def start_race(self, buttons: Button = Button.A) -> None:
        """Load the snapshot and run to the first frame the kart can move, then hold these buttons."""
        self.load_snapshot()
        self.hold(Button.A)
        while self.kart.speed() <= 0:
            self.advance()
        self.hold(buttons)

    def open_window(self) -> None:
        """Show the game in a window. It then runs at normal speed, so this is only for watching."""
        self._window = GameWindow()

    def label_window(self, text: str) -> None:
        """Show a small line of text over the game in the window, if one is open."""
        if self._window:
            self._window.set_label(text)

    def window_closed(self) -> bool:
        return self._window is not None and self._window.closed

    def hold(self, buttons: Button) -> None:
        """Hold exactly these buttons until told otherwise."""
        self._emulator.input.keypad_update(int(buttons))
        self._held = buttons

    def advance(self, frames: int = 1, draw: bool = False) -> None:
        """Run the game forward. Skipping the drawing is much faster."""
        for _ in range(frames):
            if self._window:
                self._advance_in_window(self._window)
            else:
                if not draw:
                    self._emulator.skip_next_frame()
                self._emulator.cycle(with_joystick=False)

    def _advance_in_window(self, window: GameWindow) -> None:
        self._emulator.cycle(with_joystick=False)
        window.show(self._emulator.screenshot(), self._held, self.kart.speed_fraction())
        self._next_frame_time = max(self._next_frame_time, time.perf_counter() - WATCH_FRAME_SECONDS)
        self._next_frame_time += WATCH_FRAME_SECONDS
        time.sleep(max(0.0, self._next_frame_time - time.perf_counter()))

    def tap(self, button: Button) -> None:
        """Press a button for one frame, then let go."""
        self.hold(button)
        self.advance(draw=True)
        self.hold(Button.NONE)

    def screenshot(self, path: FilePath) -> None:
        """Save both screens as an image. Draws one frame first, so the image is current."""
        self.advance(draw=True)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._emulator.screenshot().save(path)


def delete_emulator_save(rom: FilePath = ROM) -> None:
    """Delete the save the emulator keeps next to the Python install, named after the game.

    An old one can change the menus, so delete it before starting a Game that goes through them.
    """
    Path(sys.base_prefix, Path(rom).stem + ".dsv").unlink(missing_ok=True)
