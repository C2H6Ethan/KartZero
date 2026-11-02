"""Drive the menus to a Figure-8 Circuit time trial using my own save, then snapshot it.

The snapshot (a savestate) is what training reloads to restart a run. It stays on
this PC: game/ and savestates/ are ignored by git. Writes a screenshot next to it to
check by eye. Put a copy of your save, with every kart unlocked, at game/mkds.sav first.
"""

import sys
from pathlib import Path

from desmume.controls import Keys, keymask
from desmume.emulator import DeSmuME
from desmume.mkds.mkds import RACE_CONFIG_PTR_ADDR, RACER_PTR_ADDR

ROOT = Path(__file__).resolve().parent.parent
ROM = ROOT / "game" / "mkds.nds"
SAVE = ROOT / "game" / "mkds.sav"
STATE = ROOT / "savestates" / "f8c_time_trial.dst"
SHOT = ROOT / "savestates" / "f8c_time_trial.png"

A = keymask(Keys.KEY_A)
DOWN = keymask(Keys.KEY_DOWN)
RIGHT = keymask(Keys.KEY_RIGHT)

# (frames to wait, key to tap or 0 for none). Found by stepping through with
# boot_check.py. Picks Single Player, Time Trials, Dry Bones, ROB-BLS, Figure-8.
MENU_STEPS = [
    (420, 0), (60, A), (180, 0), (30, DOWN), (60, A), (180, A), (240, 0),  # to Select Character
    (30, DOWN), (30, DOWN), (30, RIGHT), (30, A), (240, 0),  # Dry Bones
    (30, RIGHT), (30, RIGHT), (30, RIGHT), (30, RIGHT), (30, RIGHT), (30, RIGHT),  # kart 7 of 36
    (60, A), (240, 0),  # ROB-BLS
    (60, A), (150, 0), (60, A), (150, 0),  # Mushroom Cup, Figure-8
    (30, RIGHT), (60, A), (150, 0),  # Ghost Data: None, so no human or staff ghost is in the race
    (60, A),  # OK
    (150, 0),  # into the countdown
]  # fmt: skip
FRAMES_AFTER_TAP = 30
PLAYER_CONFIG_OFFSET = 0x68
DRY_BONES = 8
ROB_BLS = 34


def run_frames(emu, count):
    for _ in range(count):
        emu.cycle(with_joystick=False)


def main():
    STATE.parent.mkdir(exist_ok=True)
    # The emulator keeps its save next to the Python install, named after the game.
    Path(sys.base_prefix, ROM.stem + ".dsv").unlink(missing_ok=True)
    emu = DeSmuME()
    emu.open(str(ROM))
    emu.volume_set(0)
    if not emu.backup.import_file(str(SAVE)):
        raise RuntimeError(f"could not import {SAVE}")
    emu.reset()

    for wait, key in MENU_STEPS:
        run_frames(emu, wait)
        if key:
            emu.input.keypad_update(key)
            emu.cycle(with_joystick=False)
            emu.input.keypad_update(0)
        run_frames(emu, FRAMES_AFTER_TAP)

    racer = emu.memory.unsigned.read_long(RACER_PTR_ADDR)
    if racer == 0:
        raise RuntimeError("no kart in memory: the menu script did not reach a race")
    config = emu.memory.unsigned.read_long(RACE_CONFIG_PTR_ADDR) + PLAYER_CONFIG_OFFSET
    character = emu.memory.unsigned.read_byte(config)
    kart = emu.memory.unsigned.read_byte(config + 4)
    if (character, kart) != (DRY_BONES, ROB_BLS):
        raise RuntimeError(f"expected Dry Bones in ROB-BLS, memory says {character}, {kart}")
    emu.savestate.save_file(str(STATE))
    emu.screenshot().save(SHOT)
    print(f"saved {STATE.name}, racer at {racer:#010x}, character {character}, kart {kart}")
    emu.destroy()


if __name__ == "__main__":
    main()
