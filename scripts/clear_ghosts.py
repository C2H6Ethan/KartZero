"""Remove every personal ghost from game/mkds.sav, so only the staff ghost remains.

The AI must never race against or learn from a human replay. Staff ghosts live in
the game itself, so they aren't affected. An empty slot is filled with 0xFF.
"""

from pathlib import Path

SAVE = Path(__file__).resolve().parent.parent / "game" / "mkds.sav"
GHOSTS_ADDR = 0xAF00
GHOST_SIZE = 0xE00
COURSE_COUNT = 32


def main():
    data = bytearray(SAVE.read_bytes())
    removed = 0
    for course in range(COURSE_COUNT):
        start = GHOSTS_ADDR + course * GHOST_SIZE
        if data[start:start + 2] != b"\xff\xff":
            removed += 1
        data[start:start + GHOST_SIZE] = b"\xff" * GHOST_SIZE
    SAVE.write_bytes(data)
    print(f"cleared {removed} ghosts")


if __name__ == "__main__":
    main()
