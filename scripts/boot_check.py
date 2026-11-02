"""Boot the game with no window and screenshot it, to see what a menu script must do.

Usage: python scripts/boot_check.py OUT_DIR [wait:KEY ...]
Example: python scripts/boot_check.py shots 360:A 110:DOWN
Each step runs for `wait` frames, presses KEY for one frame, then takes a screenshot.
"""

import sys
from pathlib import Path

from desmume.controls import Keys, keymask
from desmume.emulator import DeSmuME

ROM = Path(__file__).resolve().parent.parent / "game" / "mkds.nds"
SAVE = ROM.with_suffix(".sav")


def run_frames(emu, count):
    for _ in range(count):
        emu.cycle(with_joystick=False)


def main():
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    emu = DeSmuME()
    emu.open(str(ROM))
    emu.volume_set(0)
    if SAVE.exists():
        print("save imported:", emu.backup.import_file(str(SAVE)))
        emu.reset()

    for index, step in enumerate(sys.argv[2:] or ["360:NONE"]):
        wait, key = step.split(":")
        run_frames(emu, int(wait))
        if key != "NONE":
            emu.input.keypad_update(keymask(getattr(Keys, f"KEY_{key}")))
            emu.cycle(with_joystick=False)
            emu.input.keypad_update(0)
        run_frames(emu, 30)
        emu.screenshot().save(out_dir / f"{index:02d}_{key}.png")
        racer = emu.memory.unsigned.read_long(0x0217ACF8)
        course = emu.memory.unsigned.read_byte(0x23CDCD8)
        scene = emu.memory.unsigned.read_long(0x021759AC)
        print(f"step {index}: {step} racer={racer:#010x} course={course} scene_ptr={scene:#010x}")

    emu.destroy()


if __name__ == "__main__":
    main()
