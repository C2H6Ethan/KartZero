"""Shows the AI driving, live in a window.

It rolls the dice on the odds, just as training does, so you see what the AI is really practising.
Always picking the single likeliest button can look far worse while the odds are still spread out.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from kartzero.environment import ACTIONS, OBSERVATION_SIZE, RaceEnvironment
from kartzero.game import ROOT, Game
from kartzero.network import DriverNetwork
from kartzero.sensors import TrackSensors

LATEST_CHECKPOINT = ROOT / "checkpoints" / "latest.pt"
STUCK_DECISIONS = 300  # restart after 20 seconds with no progress, so there's always something to watch
STUCK_PROGRESS = 0.002


@dataclass
class ViewerRun:
    """One training run to show: its label on screen, its checkpoint, and its practice sticky chance."""

    label: str
    checkpoint: Path
    sticky_chance: float = 0.0


def _load_if_newer(network: DriverNetwork, checkpoint: Path, loaded_time: float) -> float:
    if not checkpoint.exists() or checkpoint.stat().st_mtime <= loaded_time:
        return loaded_time
    modified = checkpoint.stat().st_mtime
    network.load_state_dict(torch.load(checkpoint, map_location="cpu")["network"])
    return modified


def _race(game: Game, environment: RaceEnvironment, network: DriverNetwork, observation: np.ndarray) -> str:
    """Drive one race, rolling the dice on the odds. Returns how it went."""
    best_progress, best_decision = 0.0, 0
    while not game.window_closed():
        with torch.no_grad():
            odds, _ = network(torch.as_tensor(observation[None]))
        step = environment.step(int(odds.sample()))
        observation = step.observation
        if step.progress > best_progress + STUCK_PROGRESS:
            best_progress, best_decision = step.progress, step.decisions
        if step.finished:
            return f"FINISHED in {step.race_seconds:.2f}s"
        if step.out_of_time or step.decisions - best_decision >= STUCK_DECISIONS:
            return f"reached {step.progress:.1%} of the race"
    return "window closed"


def run_viewer(checkpoint: Path = LATEST_CHECKPOINT) -> None:
    """Race over and over. If the file gets replaced, the next race uses the newer AI."""
    torch.set_num_threads(1)
    network = DriverNetwork(OBSERVATION_SIZE, len(ACTIONS))
    loaded_time = 0.0
    with Game() as game:
        game.open_window()
        game.load_snapshot()
        environment = RaceEnvironment(game, TrackSensors(game.track.collision_triangles()))
        race = 0
        while not game.window_closed():
            race += 1
            observation = environment.reset()
            loaded_time = _load_if_newer(network, checkpoint, loaded_time)
            print(f"viewer race {race}: {_race(game, environment, network, observation)}", flush=True)


def run_rotating_viewer(runs: list[ViewerRun]) -> None:
    """Take turns between runs, one race each, always with each run's newest AI and its own sticky chance."""
    torch.set_num_threads(1)
    network = DriverNetwork(OBSERVATION_SIZE, len(ACTIONS))
    with Game() as game:
        game.open_window()
        game.load_snapshot()
        environment = RaceEnvironment(game, TrackSensors(game.track.collision_triangles()))
        race = 0
        while not game.window_closed():
            run = runs[race % len(runs)]
            race += 1
            if not run.checkpoint.exists():
                continue
            network.load_state_dict(torch.load(run.checkpoint, map_location="cpu")["network"])
            environment.sticky_chance = run.sticky_chance
            game.label_window(run.label)
            observation = environment.reset()
            print(f"viewer race {race}, {run.label}: {_race(game, environment, network, observation)}", flush=True)
