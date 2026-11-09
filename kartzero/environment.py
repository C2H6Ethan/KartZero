"""One race as the AI experiences it: see the numbers, pick an action, get a score."""

import random
from dataclasses import dataclass

import numpy as np

from kartzero.game import Button, Game
from kartzero.observation import observe
from kartzero.reward import Reward
from kartzero.sensors import TrackSensors

FRAMES_PER_DECISION = 4
DECISIONS_PER_SECOND = 15
MAX_DECISIONS_PER_RACE = 3000  # 200 seconds, enough for the test driver's 2:08 race

# Every action is one pedal and one steering: ACTIONS[pedal * len(STEERINGS) + steering].
# No item button: items stay off until we beat 1:27.832. No B button either: with reverse
# available, the first AI learned to turn around and drive the whole track backwards.
PEDALS = (Button.NONE, Button.A, Button.A | Button.R)
STEERINGS = (Button.NONE, Button.LEFT, Button.RIGHT)
ACTIONS = tuple(pedal | steering for pedal in PEDALS for steering in STEERINGS)
OBSERVATION_SIZE = 36
COUNTDOWN_INPUT = OBSERVATION_SIZE - 1  # the last input: seconds until GO


@dataclass
class Step:
    """What happened after one decision."""

    observation: np.ndarray
    reward: float
    finished: bool  # crossed the line, so no more points are possible
    out_of_time: bool  # stopped by the time limit, so more points would have been possible
    progress: float
    decisions: int
    # Whether each half of the pick decided nothing, because something else chose what was held:
    # the sticky dice, or a hold the AI picked earlier that hasn't run out yet.
    pedal_ignored: bool = False
    steering_ignored: bool = False
    boost_started: bool = False
    # The game's own time: its clock during the race, its official time at the finish. It reads 0
    # through the countdown, so it's about 3 s less than decisions / DECISIONS_PER_SECOND.
    race_seconds: float = 0.0

    @property
    def pick_ignored(self) -> bool:
        """The whole pick decided nothing."""
        return self.pedal_ignored and self.steering_ignored


class RaceEnvironment:
    """Runs races one decision at a time. It knows nothing about how the AI learns.

    With a `sticky_chance` above 0, each decision has that chance of repeating the last button
    instead of the pick. That makes long holds, like the ones a mini-turbo needs, far more common
    in practice. Test laps use 0.
    """

    def __init__(self, game: Game, sensors: TrackSensors, sticky_chance: float = 0.0, seed: int = 0) -> None:
        self._game = game
        self._sensors = sensors
        self._reward = Reward()
        self._random = random.Random(seed)
        self.sticky_chance = sticky_chance
        self._progress = 0.0
        self._decisions = 0
        self._last_action = 0  # Button.NONE
        self._boost = 0

    def reset(self) -> np.ndarray:
        """Start a new race in the countdown and return the first numbers, read after the snapshot has loaded."""
        self._game.start_countdown()
        self._progress = self._game.kart.race_progress()
        self._decisions = 0
        self._last_action = 0
        self._boost = self._game.kart.boost_time_left()
        return self._observe(Button.NONE)

    def step(self, action: int) -> Step:
        """Hold one action for a decision's worth of frames, or the last one if the sticky dice say so."""
        # Roll every time, even at 0, so the dice stay in step whatever the chance is.
        pick_ignored = self._random.random() < self.sticky_chance
        if pick_ignored:
            action = self._last_action
        self._last_action = action
        buttons = ACTIONS[action]
        self._game.hold(buttons)
        self._game.advance(FRAMES_PER_DECISION)
        self._decisions += 1

        progress = self._game.kart.race_progress()
        reward = self._reward.score(self._progress, progress)
        self._progress = progress
        finished = self._game.kart.has_finished()
        boost = self._game.kart.boost_time_left()
        boost_started = boost > self._boost  # the timer only counts down, so a rise is a new boost
        self._boost = boost
        return Step(
            observation=self._observe(buttons),
            reward=reward,
            finished=finished,
            out_of_time=not finished and self._decisions >= MAX_DECISIONS_PER_RACE,
            progress=progress,
            decisions=self._decisions,
            pedal_ignored=pick_ignored,
            steering_ignored=pick_ignored,
            boost_started=boost_started,
            race_seconds=self._game.kart.finish_time() if finished else self._game.kart.race_clock(),
        )

    def _observe(self, buttons: Button) -> np.ndarray:
        return observe(self._game.kart, self._game.track, self._sensors, buttons, self._game.seconds_until_go())
