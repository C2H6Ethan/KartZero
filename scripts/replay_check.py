"""Check that the same snapshot and the same buttons always give the same result.

Run it twice: the end position should also match between separate runs.
Run make_start_state.py first.
"""

import random

from kartzero.game import Button, Game

RUNS = 4
DECISIONS = 300
FRAMES_PER_DECISION = 4
CHOICES = [
    Button.A,
    Button.A | Button.LEFT,
    Button.A | Button.RIGHT,
    Button.A | Button.R,
    Button.A | Button.R | Button.LEFT,
    Button.A | Button.R | Button.RIGHT,
]


def run(game, plan):
    game.start_race()
    trace = []
    for buttons in plan:
        game.hold(buttons)
        game.advance(FRAMES_PER_DECISION)
        trace.append(game.kart.position())
    return trace


def main():
    plan = random.Random(1).choices(CHOICES, k=DECISIONS)
    with Game() as game:
        traces = [run(game, plan) for _ in range(RUNS)]
    print(f"\nreplays identical: {all(trace == traces[0] for trace in traces)}")
    print(f"end position: {traces[0][-1]}")


if __name__ == "__main__":
    main()
