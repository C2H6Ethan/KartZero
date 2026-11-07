"""Official test laps: no exploration, always the likeliest button, so the same AI always drives the same lap."""

from multiprocessing import Queue
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np
import torch

from kartzero.environment import RaceEnvironment, Step
from kartzero.game import Game
from kartzero.holds import race_for
from kartzero.network import load_driver
from kartzero.sensors import TrackSensors
from kartzero.versions import LapResult, VersionStore


class Environment(Protocol):
    def reset(self) -> np.ndarray: ...

    def step(self, action: int) -> Step: ...


def drive_test_lap(network: Any, environment: Environment) -> LapResult:
    """Drive one race, always choosing the likeliest action, and count its mini-turbos.
    It ends at the finish or the time limit."""
    observation = environment.reset()
    mini_turbos = 0
    while True:
        with torch.no_grad():
            odds, _ = network(torch.as_tensor(observation[None]))
        step = environment.step(int(odds.probs.argmax()))
        observation = step.observation
        mini_turbos += step.boost_started  # every boost is a mini-turbo: items are off
        if step.finished or step.out_of_time:
            return LapResult(step.finished, step.race_seconds, step.progress, mini_turbos)


def run_lap_tester(versions_to_test: "Queue[Optional[Path]]", run_folder: Path) -> None:
    """Test every version that hasn't been tested yet, then wait for new ones. Runs in its own process."""
    torch.set_num_threads(1)
    store = VersionStore(run_folder)
    with Game() as game:
        game.load_snapshot()
        race = RaceEnvironment(game, TrackSensors(game.track.collision_triangles()))

        def test(file: Path) -> None:
            network = load_driver(file)
            result = drive_test_lap(network, race_for(network, race))
            store.save_test_lap(file, result)
            print(f"test lap {file.name}: {result.description()}, {result.mini_turbos} mini-turbos", flush=True)

        tested = store.test_laps()
        for version in store.all():
            if version.file.name not in tested:
                test(version.file)
        while (file := versions_to_test.get()) is not None:
            test(file)
