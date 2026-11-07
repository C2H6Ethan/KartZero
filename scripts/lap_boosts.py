"""Drive a run's saved versions on test laps (likeliest action, no exploration) and count their boosts.

Usage: python scripts/lap_boosts.py RUN [--every N]
RUN is the end of a run folder's name, like "holds". Prints one line per tested version.
"""

import argparse
import json
from pathlib import Path

import torch

from kartzero.environment import RaceEnvironment
from kartzero.game import ROOT, Game
from kartzero.holds import make_environment, sizes
from kartzero.network import DriverNetwork
from kartzero.sensors import TrackSensors
from kartzero.settings import Settings
from kartzero.versions import VersionStore


def load_settings(run_folder: Path) -> Settings:
    saved = json.loads((run_folder / "settings.json").read_text())
    known = Settings.__dataclass_fields__
    return Settings(**{key: value for key, value in saved.items() if key in known})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--every", type=int, default=1, help="test only every Nth version")
    arguments = parser.parse_args()
    run_folder = next(folder for folder in sorted((ROOT / "runs").iterdir()) if folder.name.endswith(arguments.run))
    settings = load_settings(run_folder)
    inputs, actions, _ = sizes(settings)
    network = DriverNetwork(inputs, actions)
    with Game() as game:
        game.load_snapshot()
        environment = make_environment(RaceEnvironment(game, TrackSensors(game.track.collision_triangles())), settings)
        for version in VersionStore(run_folder).all()[:: arguments.every]:
            network.load_state_dict(torch.load(version.file, map_location="cpu")["network"])
            observation = environment.reset()
            boosts, highest, drifting = 0, 0, 0
            while True:
                with torch.no_grad():
                    odds, _ = network(torch.as_tensor(observation[None]))
                step = environment.step(int(odds.probs.argmax()))
                observation = step.observation
                boosts += step.boost_started
                highest = max(highest, game.kart.mini_turbo_count())
                drifting += game.kart.mini_turbo_count() > 0
                if step.finished or step.out_of_time:
                    break
            time = f"{step.race_seconds:7.2f}s" if step.finished else f"reached {step.progress:.1%}"
            print(
                f"{version.file.name}  {time}  boosts {boosts}  highest count {highest}  "
                f"drifting {drifting / step.decisions:.1%} of decisions",
                flush=True,
            )


if __name__ == "__main__":
    main()
