"""Drive the menus to a Figure-8 Circuit time trial using my own save, then snapshot it.

The snapshot (a savestate) is what training reloads to restart a run. It stays on
this PC: game/ and savestates/ are ignored by git. Writes a screenshot next to it to
check by eye. Put a copy of your save, with every kart unlocked, at game/mkds.sav first.
"""

from kartzero.game import FRAMES_BEFORE_GO, SNAPSHOT, Button, Game, delete_emulator_save

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
    (60, Button.A),  # OK, then into the countdown
]  # fmt: skip
FRAMES_AFTER_TAP = 30
MAX_FRAMES_TO_GO = 1200  # from the last menu tap; it took 407 when measured
DRY_BONES = 8
ROB_BLS = 34


def save_in_the_countdown(game: Game) -> None:
    """Save FRAMES_BEFORE_GO frames before GO, with nothing held, so the AI sees the countdown.

    GO is the first frame the race clock runs, and the kart can move on it if A is already held.
    Nothing may be held before the snapshot: holding A too early in the countdown makes the kart
    stall until 2.40 s after GO, which is what our first snapshot did by mistake.
    """
    game.hold(Button.NONE)
    game.save_snapshot()
    frames_to_go = 0
    while not (game.kart.is_loaded() and game.kart.race_clock() > 0):
        game.advance()
        frames_to_go += 1
        if frames_to_go > MAX_FRAMES_TO_GO:
            raise RuntimeError("the race never started: the menu script did not reach a race")
    if frames_to_go <= FRAMES_BEFORE_GO:
        raise RuntimeError(f"the menus reached the race only {frames_to_go} frames before GO")
    game.load_snapshot()
    game.advance(frames_to_go - FRAMES_BEFORE_GO)
    game.save_snapshot()
    game.start_countdown()
    game.advance(FRAMES_BEFORE_GO - 1)
    if game.kart.race_clock() != 0:
        raise RuntimeError("the race clock started before FRAMES_BEFORE_GO")
    game.advance()
    if game.kart.race_clock() <= 0:
        raise RuntimeError("the race clock didn't start at FRAMES_BEFORE_GO")
    game.load_snapshot()


def main() -> None:
    delete_emulator_save()
    with Game() as game:
        game.load_save()
        for wait, button in MENU_STEPS:
            game.advance(wait, draw=True)
            if button:
                game.tap(button)
            game.advance(FRAMES_AFTER_TAP, draw=True)

        save_in_the_countdown(game)
        if not game.kart.is_loaded():
            raise RuntimeError("no kart in memory: the menu script did not reach a race")
        if (game.kart.driver_id(), game.kart.kart_id()) != (DRY_BONES, ROB_BLS):
            raise RuntimeError(f"expected Dry Bones in ROB-BLS, memory says {game.kart.driver_id()}, {game.kart.kart_id()}")
        game.screenshot(SNAPSHOT.with_suffix(".png"))
        print(f"saved {SNAPSHOT.name}: driver {game.kart.driver_id()}, kart {game.kart.kart_id()}")


if __name__ == "__main__":
    main()
