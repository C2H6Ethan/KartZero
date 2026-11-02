"""Drive the menus to a Figure-8 Circuit time trial using my own save, then snapshot it.

The snapshot (a savestate) is what training reloads to restart a run. It stays on
this PC: game/ and savestates/ are ignored by git. Writes a screenshot next to it to
check by eye. Put a copy of your save, with every kart unlocked, at game/mkds.sav first.
"""

from kartzero.game import SNAPSHOT, Button, Game, delete_emulator_save

# (frames to wait, button to tap). Found by stepping through with boot_check.py.
# Picks Single Player, Time Trials, Dry Bones, ROB-BLS, Figure-8.
MENU_STEPS = [
    (420, Button.NONE), (60, Button.A), (180, Button.NONE), (30, Button.DOWN), (60, Button.A),
    (180, Button.A), (240, Button.NONE),  # to Select Character
    (30, Button.DOWN), (30, Button.DOWN), (30, Button.RIGHT), (30, Button.A), (240, Button.NONE),  # Dry Bones
    (30, Button.RIGHT), (30, Button.RIGHT), (30, Button.RIGHT),
    (30, Button.RIGHT), (30, Button.RIGHT), (30, Button.RIGHT),  # kart 7 of 36
    (60, Button.A), (240, Button.NONE),  # ROB-BLS
    (60, Button.A), (150, Button.NONE), (60, Button.A), (150, Button.NONE),  # Mushroom Cup, Figure-8
    (30, Button.RIGHT), (60, Button.A), (150, Button.NONE),  # Ghost Data: None, so no ghost is in the race
    (60, Button.A),  # OK
    (150, Button.NONE),  # into the countdown
]  # fmt: skip
FRAMES_AFTER_TAP = 30
DRY_BONES = 8
ROB_BLS = 34


def main():
    delete_emulator_save()
    with Game() as game:
        game.load_save()
        for wait, button in MENU_STEPS:
            game.advance(wait, draw=True)
            if button:
                game.tap(button)
            game.advance(FRAMES_AFTER_TAP, draw=True)

        if not game.kart.is_loaded():
            raise RuntimeError("no kart in memory: the menu script did not reach a race")
        if (game.kart.driver_id(), game.kart.kart_id()) != (DRY_BONES, ROB_BLS):
            raise RuntimeError(f"expected Dry Bones in ROB-BLS, memory says {game.kart.driver_id()}, {game.kart.kart_id()}")
        game.save_snapshot()
        game.screenshot(SNAPSHOT.with_suffix(".png"))
        print(f"saved {SNAPSHOT.name}: driver {game.kart.driver_id()}, kart {game.kart.kart_id()}")


if __name__ == "__main__":
    main()
