"""Check that the game's progress data behaves well enough to build a reward on.

Drives forward, reverses, reverses over the start line, then lets a simple test
driver complete real laps by steering at the track's checkpoints. The AI never uses
this driver. Run make_start_state.py first.
"""

from dataclasses import dataclass

from kartzero.game import Button, Game
from kartzero.test_driver import TestDriver

FRAMES_PER_SAMPLE = 4
SAMPLES_PER_SECOND = 15
MAX_SECONDS = 180


@dataclass
class Sample:
    race_progress: float
    lap_progress: float
    checkpoint: int
    lap: int


def take_sample(kart):
    return Sample(kart.race_progress(), kart.lap_progress(), kart.checkpoint(), kart.lap())


def show(title, game, seconds):
    print(f"\n{title}")
    for second in range(seconds):
        game.advance(60)
        now = take_sample(game.kart)
        print(
            f"  t={second + 1:2d}s speed={game.kart.speed():6.2f} race={now.race_progress:8.4f} "
            f"lap_progress={now.lap_progress:8.4f} checkpoint={now.checkpoint:3d} lap={now.lap}"
        )


def drive_laps(game):
    count = game.track.checkpoint_count()
    print(f"\nTest driver: {count} checkpoints on this track")
    game.start_race()
    driver = TestDriver(game)
    history = [take_sample(game.kart)]
    for step in range(MAX_SECONDS * SAMPLES_PER_SECOND):
        driver.drive()
        game.advance(FRAMES_PER_SAMPLE)
        now = take_sample(game.kart)
        before = history[-1]
        if now.lap != before.lap:
            print(
                f"  lap {before.lap} -> {now.lap} at {step / SAMPLES_PER_SECOND:.1f}s, "
                f"race {before.race_progress:.4f} -> {now.race_progress:.4f}"
            )
        history.append(now)
        if now.lap >= 3:
            break

    changes = [after.race_progress - before.race_progress for before, after in zip(history, history[1:])]
    print(f"  samples {len(changes)}, end race {history[-1].race_progress:.4f}, lap {history[-1].lap}")
    print(f"  biggest rise {max(changes):.5f}, typical rise {sorted(changes)[len(changes) // 2]:.5f}")
    print(f"  drops {sum(change < 0 for change in changes)}, biggest drop {min(changes):.5f}")
    print(f"  checkpoints seen {len({sample.checkpoint for sample in history})} of {count}")


def main():
    with Game() as game:
        game.start_race()
        show("Accelerator: race progress should rise", game, 5)
        game.hold(Button.B)
        show("Then hold B: progress should fall once the kart reverses", game, 8)

        game.start_race(Button.B)
        show("From the start, reverse over the start line: look for jumps", game, 6)
        game.hold(Button.A)
        show("Then drive forward over the line again", game, 6)

        drive_laps(game)


if __name__ == "__main__":
    main()
