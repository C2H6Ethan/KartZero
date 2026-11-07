"""Official test laps: no exploration, always the likeliest button, so the same AI always drives the same lap."""

from multiprocessing import Queue
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np
import torch

from kartzero.environment import RaceEnvironment, Step
from kartzero.holds import make_environment, sizes
from kartzero.game import Game
from kartzero.network import DriverNetwork
from kartzero.settings import Settings
from kartzero.sensors import TrackSensors
from kartzero.versions import LapResult, VersionStore


class Environment(Protocol):
    def reset(self) -> np.ndarray: ...

    def step(self, action: int) -> Step: ...


def drive_test_lap(network: Any, environment: Environment) -> LapResult:
    """Drive one race, always choosing the likeliest action. It ends at the finish or the time limit."""
    observation = environment.reset()
    while True:
        with torch.no_grad():
            odds, _ = network(torch.as_tensor(observation[None]))
        step = environment.step(int(odds.probs.argmax()))
        observation = step.observation
        if step.finished or step.out_of_time:
            return LapResult(step.finished, step.race_seconds, step.progress)


def run_lap_tester(
    versions_to_test: "Queue[Optional[Path]]", run_folder: Path, settings: Optional[Settings] = None
) -> None:
    """Test every version that hasn't been tested yet, then wait for new ones. Runs in its own process."""
    torch.set_num_threads(1)
    store = VersionStore(run_folder)
    settings = settings or Settings()
    inputs, actions, _ = sizes(settings)
    network = DriverNetwork(inputs, actions)
    with Game() as game:
        game.load_snapshot()
        environment = make_environment(RaceEnvironment(game, TrackSensors(game.track.collision_triangles())), settings)

        def test(file: Path) -> None:
            network.load_state_dict(torch.load(file, map_location="cpu")["network"])
            result = drive_test_lap(network, environment)
            store.save_test_lap(file, result)
            print(f"test lap {file.name}: {result.description()}", flush=True)

        tested = store.test_laps()
        for version in store.all():
            if version.file.name not in tested:
                test(version.file)
        while (file := versions_to_test.get()) is not None:
            test(file)
