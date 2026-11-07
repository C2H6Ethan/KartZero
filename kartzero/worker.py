"""Runs one emulator in its own process. It doesn't load PyTorch, which keeps it light."""

from multiprocessing.connection import Connection
from pathlib import Path
from typing import Optional

from kartzero.archive import StateArchive
from kartzero.environment import RaceEnvironment
from kartzero.game import Game
from kartzero.holds import make_environment
from kartzero.sensors import TrackSensors
from kartzero.settings import Settings


def run_worker(connection: Connection, settings: Settings, seed: int = 0, archive_folder: Optional[Path] = None) -> None:
    """Receive an action, send back what happened. A new race starts by itself when one ends."""
    archive = None
    if settings.archive_start_chance > 0 and archive_folder is not None:
        archive = StateArchive(
            archive_folder, settings.archive_capacity, settings.archive_decisions_between_saves, seed
        )
    try:
        with Game() as game:
            game.load_snapshot()
            race = RaceEnvironment(
                game,
                TrackSensors(game.track.collision_triangles()),
                settings.sticky_chance,
                seed,
                settings.sticky_per_button,
                archive,
                settings.archive_start_chance,
                settings.archive_race_decisions,
                settings.archive_explore_decisions,
            )
            environment = make_environment(race, settings)
            connection.send(environment.reset())
            while True:
                action = connection.recv()
                if action is None:
                    return
                step = environment.step(action)
                race_over = step.finished or step.out_of_time
                connection.send((step, environment.reset() if race_over else step.observation))
    finally:
        if archive is not None:
            archive.delete()  # saved game states never stay on disk after training
