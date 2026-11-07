"""Drive every saved version of a run again, and record its test lap with the game's official time.

Test laps used to be timed as decisions / 15, which leaves out the 2.4 s the game's clock runs in
the countdown. The old file is kept as test_laps_decision_clock.csv. Replays are identical, so the
same version drives the same lap, and only the time changes. Safe to stop and run again.

Usage: python scripts/retime_test_laps.py RUN [RUN ...]    (names as in watch.py --list)
Don't run it on a run that is still training: its lap tester is writing to the same file.
"""

import argparse

import torch

from kartzero.environment import ACTIONS, OBSERVATION_SIZE, RaceEnvironment
from kartzero.game import Game
from kartzero.network import DriverNetwork
from kartzero.runs import TrainingRun, find_run
from kartzero.sensors import TrackSensors
from kartzero.test_lap import drive_test_lap
from kartzero.versions import TEST_LAPS_FILE

OLD_TEST_LAPS_FILE = "test_laps_decision_clock.csv"


def retime(run: TrainingRun, environment: RaceEnvironment, network: DriverNetwork) -> None:
    folder = run.folder / "versions"
    old = folder / OLD_TEST_LAPS_FILE
    if not old.exists():
        (folder / TEST_LAPS_FILE).rename(old)  # only on the first go, so a second go carries on
    done = run.versions.test_laps()
    versions = run.versions.all()
    for number, version in enumerate(versions, 1):
        if version.file.name in done:
            continue
        network.load_state_dict(torch.load(version.file, map_location="cpu")["network"])
        result = drive_test_lap(network, environment)
        run.versions.save_test_lap(version.file, result)
        print(f"{run.name} {number}/{len(versions)} {version.file.name}: {result.description()}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    arguments = parser.parse_args()
    runs = [find_run(name) for name in arguments.runs]

    torch.set_num_threads(1)
    network = DriverNetwork(OBSERVATION_SIZE, len(ACTIONS))
    with Game() as game:
        game.load_snapshot()
        environment = RaceEnvironment(game, TrackSensors(game.track.collision_triangles()))
        for run in runs:
            retime(run, environment, network)


if __name__ == "__main__":
    main()
