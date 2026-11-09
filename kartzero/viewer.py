"""Shows the AI driving, live in a window.

It rolls the dice on the odds, just as training does, so you see what the AI is really practising.
Always picking the single likeliest button can look far worse while the odds are still spread out.
With `test_lap`, it picks the likeliest button instead, exactly like an official test lap, so it
replays that version's test lap.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from kartzero.environment import RaceEnvironment
from kartzero.game import ROOT, Game
from kartzero.holds import race_for
from kartzero.network import DriverNetwork, load_driver
from kartzero.sensors import TrackSensors
from kartzero.test_lap import Environment

LATEST_CHECKPOINT = ROOT / "checkpoints" / "latest.pt"
STUCK_DECISIONS = 300  # restart after 20 seconds with no progress, so there's always something to watch
STUCK_PROGRESS = 0.002


@dataclass
class ViewerRun:
    """One training run to show: its label on screen, its checkpoint, and its practice sticky chance."""

    label: str
    checkpoint: Path
    sticky_chance: float = 0.0


def _race(game: Game, environment: Environment, network: DriverNetwork, test_lap: bool = False) -> str:
    """Drive one race, rolling the dice on the odds, or with the likeliest buttons for a test lap.
    Returns how it went."""
    observation: np.ndarray = environment.reset()
    best_progress, best_decision = 0.0, 0
    while not game.window_closed():
        with torch.no_grad():
            odds, _ = network(torch.as_tensor(observation[None]))
        step = environment.step(int(odds.probs.argmax() if test_lap else odds.sample()))
        observation = step.observation
        if step.progress > best_progress + STUCK_PROGRESS:
            best_progress, best_decision = step.progress, step.decisions
        if step.finished:
            return f"FINISHED in {step.race_seconds:.2f}s"
        if step.out_of_time or (not test_lap and step.decisions - best_decision >= STUCK_DECISIONS):
            return f"reached {step.progress:.1%} of the race"
    return "window closed"


def run_viewer(brain: Callable[[], Path] = lambda: LATEST_CHECKPOINT, test_lap: bool = False) -> None:
    """Race over and over. `brain` says which saved AI to drive, and it's asked again before every
    race, so a newer file, or a file that was replaced, is picked up for the next race."""
    torch.set_num_threads(1)
    network: Optional[DriverNetwork] = None
    loaded: Optional[tuple[Path, float]] = None  # which file is loaded, and when it was written
    with Game() as game:
        game.open_window()
        game.load_snapshot()
        race = RaceEnvironment(game, TrackSensors(game.track.collision_triangles()))
        number = 0
        while not game.window_closed():
            file = brain()
            if file.exists() and loaded != (file, file.stat().st_mtime):
                loaded = (file, file.stat().st_mtime)
                network = load_driver(file)
                game.label_window(_describe(file))
            if network is None:  # nothing saved yet
                game.advance(60)
                continue
            number += 1
            print(f"viewer race {number}, {_describe(file)}: {_race(game, race_for(network, race), network, test_lap)}", flush=True)


def _describe(file: Path) -> str:
    """A short name for a saved AI: its run and training time for a saved version, else the file name."""
    if file.parent.name == "versions":
        return f"{file.parent.parent.name}  {file.name.split('_')[0]}"
    return file.name


def run_rotating_viewer(runs: list[ViewerRun]) -> None:
    """Take turns between runs, one race each, always with each run's newest AI and its own sticky chance."""
    torch.set_num_threads(1)
    with Game() as game:
        game.open_window()
        game.load_snapshot()
        race = RaceEnvironment(game, TrackSensors(game.track.collision_triangles()))
        number = 0
        while not game.window_closed():
            run = runs[number % len(runs)]
            number += 1
            if not run.checkpoint.exists():
                continue
            network = load_driver(run.checkpoint)
            race.sticky_chance = run.sticky_chance
            game.label_window(run.label)
            print(f"viewer race {number}, {run.label}: {_race(game, race_for(network, race), network)}", flush=True)
