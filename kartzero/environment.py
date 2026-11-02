"""One race as the AI experiences it: see the numbers, pick an action, get a score."""

from dataclasses import dataclass

import numpy as np

from kartzero.game import Button, Game
from kartzero.observation import observe
from kartzero.reward import Reward
from kartzero.sensors import TrackSensors

FRAMES_PER_DECISION = 4
DECISIONS_PER_SECOND = 15
MAX_DECISIONS_PER_RACE = 3000  # 200 seconds, enough for the test driver's 2:08 race

# No item button: items stay off until we beat 1:27.832. No B button either: with reverse
# available, the first AI learned to turn around and drive the whole track backwards.
ACTIONS = (
    Button.NONE,
    Button.LEFT,
    Button.RIGHT,
    Button.A,
    Button.A | Button.LEFT,
    Button.A | Button.RIGHT,
    Button.A | Button.R,
    Button.A | Button.R | Button.LEFT,
    Button.A | Button.R | Button.RIGHT,
)
OBSERVATION_SIZE = 35


@dataclass
class Step:
    """What happened after one decision."""

    observation: np.ndarray
    reward: float
    finished: bool  # crossed the line, so no more points are possible
    out_of_time: bool  # stopped by the time limit, so more points would have been possible
    progress: float
    decisions: int


class RaceEnvironment:
    """Runs races one decision at a time. It knows nothing about how the AI learns."""

    def __init__(self, game: Game, sensors: TrackSensors) -> None:
        self._game = game
        self._sensors = sensors
        self._reward = Reward()
        self._progress = 0.0
        self._decisions = 0

    def reset(self) -> np.ndarray:
        """Start a new race and return the first numbers, read after the snapshot has loaded."""
        self._game.start_race(Button.NONE)
        self._progress = self._game.kart.race_progress()
        self._decisions = 0
        return self._observe(Button.NONE)

    def step(self, action: int) -> Step:
        """Hold one action for a decision's worth of frames."""
        buttons = ACTIONS[action]
        self._game.hold(buttons)
        self._game.advance(FRAMES_PER_DECISION)
        self._decisions += 1

        progress = self._game.kart.race_progress()
        reward = self._reward.score(self._progress, progress)
        self._progress = progress
        finished = self._game.kart.has_finished()
        return Step(
            observation=self._observe(buttons),
            reward=reward,
            finished=finished,
            out_of_time=not finished and self._decisions >= MAX_DECISIONS_PER_RACE,
            progress=progress,
            decisions=self._decisions,
        )

    def _observe(self, buttons: Button) -> np.ndarray:
        return observe(self._game.kart, self._game.track, self._sensors, buttons)
