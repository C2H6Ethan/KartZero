"""Drive a run's saved versions on test laps (likeliest action, no exploration) and count their mini-turbos.

Usage: python scripts/lap_boosts.py RUN [--every N] [--between FIRST LAST]
RUN is a run's folder name or the end of it, like "holds". Prints one line per tested version.
"""

import argparse

import torch

from kartzero.environment import RaceEnvironment
from kartzero.game import Game
from kartzero.holds import race_for
from kartzero.network import load_driver
from kartzero.runs import find_run
from kartzero.sensors import TrackSensors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--every", type=int, default=1, help="test only every Nth version")
    parser.add_argument("--between", type=float, nargs=2, metavar=("FIRST", "LAST"), help="only versions in this range of training minutes")
    arguments = parser.parse_args()
    run = find_run(arguments.run)
    with Game() as game:
        game.load_snapshot()
        race = RaceEnvironment(game, TrackSensors(game.track.collision_triangles()))
        versions = run.versions.all()
        if arguments.between:
            first, last = arguments.between
            versions = [version for version in versions if first <= version.minutes <= last]
        for version in versions[:: arguments.every]:
            network = load_driver(version.file)
            environment = race_for(network, race)
            observation = environment.reset()
            boosts, highest_count, drifting = 0, 0, 0
            while True:
                with torch.no_grad():
                    odds, _ = network(torch.as_tensor(observation[None]))
                step = environment.step(int(odds.probs.argmax()))
                observation = step.observation
                count = game.kart.mini_turbo_count()
                boosts += step.boost_started
                highest_count = max(highest_count, count)
                drifting += count > 0
                if step.finished or step.out_of_time:
                    break
            result = f"{step.race_seconds:7.2f}s" if step.finished else f"reached {step.progress:.1%}"
            print(
                f"{version.file.name}  {result}  boosts {boosts}  highest count {highest_count}  "
                f"drifting {drifting / step.decisions:.1%} of decisions",
                flush=True,
            )


if __name__ == "__main__":
    main()
