"""Runs one emulator in its own process. It doesn't load PyTorch, which keeps it light."""

from multiprocessing.connection import Connection

from kartzero.environment import RaceEnvironment
from kartzero.game import Game
from kartzero.holds import HoldEnvironment
from kartzero.sensors import TrackSensors
from kartzero.settings import Settings


def run_worker(connection: Connection, settings: Settings, seed: int = 0) -> None:
    """Receive an action, send back what happened. A new race starts by itself when one ends."""
    with Game() as game:
        game.load_snapshot()
        race = RaceEnvironment(game, TrackSensors(game.track.collision_triangles()), settings.sticky_chance, seed)
        environment = HoldEnvironment(race) if settings.hold_lengths else race
        connection.send(environment.reset())
        while True:
            action = connection.recv()
            if action is None:
                return
            step = environment.step(action)
            race_over = step.finished or step.out_of_time
            connection.send((step, environment.reset() if race_over else step.observation))
