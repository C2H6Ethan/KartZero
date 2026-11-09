"""Check the AI's inputs on the real game: right length, sensible sizes, and how much they cost.

Races with the test driver, prints the smallest and largest value of each input, and flags
any input that never changes or goes far outside -1 to 1. Run make_start_state.py first.
"""

import time

import numpy as np

from kartzero.game import Button, Game
from kartzero.observation import BUTTONS, CHECKPOINTS_AHEAD, observe
from kartzero.sensors import RAY_ANGLES, TrackSensors
from kartzero.test_driver import TestDriver

FRAMES_PER_DECISION = 4
TIMING_DECISIONS = 3000
LARGEST_EXPECTED = 1.5


def input_names() -> list[str]:
    names = ["speed", "drift_angle", "turning_rate", "mini_turbo", "boost", "off_road", "on_ground"]
    names += [f"button_{button.name}" for button in BUTTONS]
    names += [f"wall_{round(np.degrees(angle))}" for angle in RAY_ANGLES]
    names += [f"edge_{round(np.degrees(angle))}" for angle in RAY_ANGLES]
    for ahead in CHECKPOINTS_AHEAD:
        names += [f"checkpoint{ahead}_distance", f"checkpoint{ahead}_angle"]
    return names


def main() -> None:
    names = input_names()
    with Game() as game:
        game.load_snapshot()
        sensors = TrackSensors(game.track.collision_triangles())
        game.start_race()
        driver = TestDriver(game)
        rows = []
        decisions = 0
        while not game.kart.has_finished():
            driver.drive()
            # Mix in drifting, so that the drift and boost inputs get used too.
            held = Button.A | Button.R if decisions % 200 < 40 else Button.A
            game.advance(FRAMES_PER_DECISION)
            rows.append(observe(game.kart, game.track, sensors, held, game.seconds_until_go()))
            decisions += 1
        values = np.array(rows)
        print(f"\n{decisions} decisions, {values.shape[1]} inputs, expected {len(names)}, type {values.dtype}")
        print(f"not-a-number values: {int(np.isnan(values).sum())}")
        for name, column in zip(names, values.T):
            notes = " NEVER CHANGES" if column.min() == column.max() else ""
            notes += " TOO LARGE" if np.abs(column).max() > LARGEST_EXPECTED else ""
            print(f"  {name:22s} {column.min():7.3f} to {column.max():7.3f}{notes}")

        game.start_race()
        start = time.perf_counter()
        for _ in range(TIMING_DECISIONS):
            game.advance(FRAMES_PER_DECISION)
            observe(game.kart, game.track, sensors, Button.A, game.seconds_until_go())
        print(f"\nSpeed with inputs built every decision: {TIMING_DECISIONS * FRAMES_PER_DECISION / (time.perf_counter() - start):.0f} frames/s")


if __name__ == "__main__":
    main()
