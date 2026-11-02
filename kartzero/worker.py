"""Runs one emulator in its own process. It doesn't load PyTorch, which keeps it light."""

from multiprocessing.connection import Connection

from kartzero.environment import RaceEnvironment
from kartzero.game import Game
from kartzero.sensors import TrackSensors


def run_worker(connection: Connection) -> None:
    """Receive an action, send back what happened. A new race starts by itself when one ends."""
    with Game() as game:
        game.load_snapshot()
        environment = RaceEnvironment(game, TrackSensors(game.track.collision_triangles()))
        connection.send(environment.reset())
        while True:
            action = connection.recv()
            if action is None:
                return
            step = environment.step(action)
            race_over = step.finished or step.out_of_time
            connection.send((step, environment.reset() if race_over else step.observation))
