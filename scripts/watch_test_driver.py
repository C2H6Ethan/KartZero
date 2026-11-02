"""Open a window and watch the test driver race all 3 laps of Figure-8 at normal speed.

Prints each lap and what the progress data does at the finish. Close the window to stop.
Run make_start_state.py first.
"""

from kartzero.game import Game
from kartzero.test_driver import TestDriver

FRAMES_PER_DECISION = 4
FRAMES_PER_SECOND = 60
MAX_SECONDS = 300
SECONDS_AFTER_FINISH = 10


def main():
    with Game() as game:
        game.open_window()
        game.start_race()
        driver = TestDriver(game)
        frames = 0
        lap = game.kart.lap()
        finish_frame = None
        while not game.window_closed() and frames < MAX_SECONDS * FRAMES_PER_SECOND:
            if finish_frame is None:
                driver.drive()
            game.advance(FRAMES_PER_DECISION, draw=True)
            frames += FRAMES_PER_DECISION

            seconds = frames / FRAMES_PER_SECOND
            progress = game.kart.race_progress()
            if game.kart.lap() != lap:
                lap = game.kart.lap()
                print(f"{seconds:6.1f}s lap {lap}, race progress {progress:.4f}", flush=True)
            if finish_frame is None and game.kart.has_finished():
                finish_frame = frames
                print(f"{seconds:6.1f}s FINISHED, race progress {progress:.4f}", flush=True)
            if finish_frame is not None:
                if frames % FRAMES_PER_SECOND == 0:
                    print(f"{seconds:6.1f}s after finish: race progress {progress:.4f}, speed {game.kart.speed():.2f}", flush=True)
                if frames - finish_frame >= SECONDS_AFTER_FINISH * FRAMES_PER_SECOND:
                    break
        if finish_frame is None:
            print(f"stopped at {frames / FRAMES_PER_SECOND:.1f}s without finishing, race progress {game.kart.race_progress():.4f}", flush=True)


if __name__ == "__main__":
    main()
