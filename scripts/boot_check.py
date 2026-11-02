"""Boot the game with no window and screenshot it, to see what a menu script must do.

Usage: python scripts/boot_check.py OUT_DIR [wait:BUTTON ...]
Example: python scripts/boot_check.py shots 360:A 110:DOWN
Each step runs for `wait` frames, taps BUTTON (or NONE), then takes a screenshot.
"""

import sys
from pathlib import Path

from kartzero.game import SAVE, Button, Game

FRAMES_AFTER_TAP = 30


def main():
    out_dir = Path(sys.argv[1])
    with Game() as game:
        if SAVE.exists():
            game.load_save()
        for index, step in enumerate(sys.argv[2:] or ["360:NONE"]):
            wait, name = step.split(":")
            game.advance(int(wait), draw=True)
            if Button[name]:
                game.tap(Button[name])
            game.advance(FRAMES_AFTER_TAP, draw=True)
            game.screenshot(out_dir / f"{index:02d}_{name}.png")
            print(f"step {index}: {step} kart loaded: {game.kart.is_loaded()}", flush=True)


if __name__ == "__main__":
    main()
