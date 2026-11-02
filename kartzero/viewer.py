"""Shows the AI driving, live in a window.

It rolls the dice on the odds, just as training does, so you see what the AI is really practising.
Always picking the single likeliest button can look far worse while the odds are still spread out.
"""

from pathlib import Path

import torch

from kartzero.environment import ACTIONS, DECISIONS_PER_SECOND, OBSERVATION_SIZE, RaceEnvironment
from kartzero.game import ROOT, Game
from kartzero.network import DriverNetwork
from kartzero.sensors import TrackSensors

LATEST_CHECKPOINT = ROOT / "checkpoints" / "latest.pt"
STUCK_DECISIONS = 300  # restart after 20 seconds with no progress, so there's always something to watch
STUCK_PROGRESS = 0.002


def _load_if_newer(network: DriverNetwork, checkpoint: Path, loaded_time: float) -> float:
    if not checkpoint.exists() or checkpoint.stat().st_mtime <= loaded_time:
        return loaded_time
    modified = checkpoint.stat().st_mtime
    network.load_state_dict(torch.load(checkpoint, map_location="cpu")["network"])
    return modified


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
            best_progress, best_decision = 0.0, 0
            while not game.window_closed():
                with torch.no_grad():
                    odds, _ = network(torch.as_tensor(observation[None]))
                step = environment.step(int(odds.sample()))
                observation = step.observation
                if step.progress > best_progress + STUCK_PROGRESS:
                    best_progress, best_decision = step.progress, step.decisions
                if step.finished:
                    print(f"viewer race {race}: FINISHED in {step.decisions / DECISIONS_PER_SECOND:.1f}s", flush=True)
                    break
                if step.out_of_time or step.decisions - best_decision >= STUCK_DECISIONS:
                    print(f"viewer race {race}: reached {step.progress:.1%} of the race", flush=True)
                    break
