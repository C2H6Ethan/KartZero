"""Count the mini-turbos on a run's test laps, for versions tested before the lap tester counted them.

It drives each version's test lap again (likeliest action, no exploration), which is the same lap
every time, saves the count with the run, and prints it with how much of the lap was spent drifting.
`watch.py --list RUN` then shows the counts. Versions that already have a count are skipped.

Usage: python scripts/lap_boosts.py RUN [--every N] [--between FIRST LAST] [--recount]
RUN is a run's folder name or the end of it, like "holds".
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
    parser.add_argument(
        "--between", type=float, nargs=2, metavar=("FIRST", "LAST"), help="only versions in this range of training minutes"
    )
    parser.add_argument("--recount", action="store_true", help="drive versions that already have a count too")
    arguments = parser.parse_args()
    run = find_run(arguments.run)
    store = run.versions
    laps = store.test_laps()
    versions = store.all()
    if arguments.between:
        first, last = arguments.between
        versions = [version for version in versions if first <= version.minutes <= last]
    versions = versions[:: arguments.every]
    if not arguments.recount:
        versions = [v for v in versions if v.file.name not in laps or laps[v.file.name].mini_turbos is None]
    print(f"{run.name}: counting {len(versions)} versions", flush=True)
    with Game() as game:
        game.load_snapshot()
        race = RaceEnvironment(game, TrackSensors(game.track.collision_triangles()))
        for version in versions:
            network = load_driver(version.file)
            environment = race_for(network, race)
            observation = environment.reset()
            mini_turbos, highest_count, drifting = 0, 0, 0
            while True:
                with torch.no_grad():
                    odds, _ = network(torch.as_tensor(observation[None]))
                step = environment.step(int(odds.probs.argmax()))
                observation = step.observation
                count = game.kart.mini_turbo_count()
                mini_turbos += step.boost_started
                highest_count = max(highest_count, count)
                drifting += count > 0
                if step.finished or step.out_of_time:
                    break
            store.save_mini_turbo_count(version.file, mini_turbos)
            result = f"{step.race_seconds:7.2f}s" if step.finished else f"reached {step.progress:.1%}"
            print(
                f"{version.file.name}  {result}  mini-turbos {mini_turbos}  highest count {highest_count}  "
                f"drifting {drifting / step.decisions:.1%} of decisions",
                flush=True,
            )


if __name__ == "__main__":
    main()
