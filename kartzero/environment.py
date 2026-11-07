"""One race as the AI experiences it: see the numbers, pick an action, get a score."""

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

from kartzero.archive import StateArchive, cell_of
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
# Every action is one pedal choice and one steering choice: ACTIONS[pedal * STEERING_CHOICES + steering].
PEDAL_CHOICES = 3  # nothing, A, A + R
STEERING_CHOICES = 3  # straight, LEFT, RIGHT


def pedal_of(action: int) -> int:
    return action // STEERING_CHOICES


def steering_of(action: int) -> int:
    return action % STEERING_CHOICES


@dataclass
class Step:
    """What happened after one decision."""

    observation: np.ndarray
    reward: float
    finished: bool  # crossed the line, so no more points are possible
    out_of_time: bool  # stopped by the time limit, so more points would have been possible
    progress: float
    decisions: int
    pick_ignored: bool = False  # the sticky dice repeated the last button, so the pick decided nothing
    # With per-button dice, each half of the pick can be repeated on its own.
    pedal_ignored: bool = False
    steering_ignored: bool = False
    boost_started: bool = False
    # The game's own time: its clock during the race, its official time at the finish. It's
    # about 2.4 s more than decisions / DECISIONS_PER_SECOND, because the clock starts in the countdown.
    race_seconds: float = 0.0
    from_archive: bool = False  # this race started from an archived moment, not the start line
    held_action: int = 0  # the action really held: the pick, or what the sticky dice kept


class RaceEnvironment:
    """Runs races one decision at a time. It knows nothing about how the AI learns.

    With a `sticky_chance` above 0, each decision has that chance of repeating the last button
    instead of the pick. That makes long holds, like the ones a mini-turbo needs, far more common
    in practice. Test laps use 0.

    With `per_button`, the pedal (nothing, A, A + R) and the steering each roll their own dice with
    the same chance, so a held pedal can outlast several steering changes. Every button is still
    treated the same way.
    """

    def __init__(
        self,
        game: Game,
        sensors: TrackSensors,
        sticky_chance: float = 0.0,
        seed: int = 0,
        per_button: bool = False,
        archive: Optional[StateArchive] = None,
        archive_start_chance: float = 0.0,
        archive_race_decisions: int = MAX_DECISIONS_PER_RACE,
        archive_explore_decisions: int = 0,
    ) -> None:
        self._game = game
        self._sensors = sensors
        self._reward = Reward()
        self._random = random.Random(seed)
        self.sticky_chance = sticky_chance
        self.per_button = per_button
        self._archive = archive
        self._archive_start_chance = archive_start_chance
        self._archive_race_decisions = archive_race_decisions
        self._archive_random = random.Random(seed + 1_000_000)  # apart from the sticky dice, so they stay in step
        self._from_archive = False
        self._archive_explore_decisions = archive_explore_decisions
        self._explore_left = 0
        self._progress = 0.0
        self._decisions = 0
        self._last_action = 0  # Button.NONE
        self._boost = 0

    def reset(self) -> np.ndarray:
        """Start a new race and return the first numbers, read after the snapshot has loaded.

        With an archive, some races start from an archived moment instead, holding what was held then.
        """
        picked = None
        if self._archive is not None and self._archive_random.random() < self._archive_start_chance:
            picked = self._archive.pick()
        if picked is None:
            self._game.start_race(Button.NONE)
            self._last_action = 0
        else:
            file, self._last_action = picked
            self._game.load_snapshot(file)
            self._game.hold(ACTIONS[self._last_action])
        self._from_archive = picked is not None
        # Go-Explore: after returning to a rare moment, explore with random buttons for a while.
        # A random length from 0 up, so the AI also gets to act straight from the moment itself.
        self._explore_left = self._archive_random.randint(0, self._archive_explore_decisions) if picked else 0
        self._progress = self._game.kart.race_progress()
        self._decisions = 0
        self._boost = self._game.kart.boost_time_left()
        return self._observe(ACTIONS[self._last_action])

    def step(self, action: int) -> Step:
        """Hold one action for a decision's worth of frames, or the last one if the sticky dice say so."""
        exploring = self._explore_left > 0
        if exploring:
            self._explore_left -= 1
            action = self._archive_random.randrange(len(ACTIONS))
        # Roll every time, even at 0, so the dice stay in step whatever the chance is.
        pedal_ignored = self._random.random() < self.sticky_chance
        steering_ignored = self._random.random() < self.sticky_chance if self.per_button else pedal_ignored
        pedal = pedal_of(self._last_action if pedal_ignored else action)
        steering = steering_of(self._last_action if steering_ignored else action)
        action = pedal * STEERING_CHOICES + steering
        if exploring:  # the AI's pick decided nothing, so it mustn't be pushed by what followed
            pedal_ignored = steering_ignored = True
        pick_ignored = pedal_ignored and steering_ignored
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
        observation = self._observe(buttons)
        if self._archive is not None and not finished:
            cell = cell_of(observation, self._game.kart.checkpoint())
            if self._archive.visit(cell):
                self._game.save_snapshot(self._archive.file_for(cell, action))
        limit = self._archive_race_decisions if self._from_archive else MAX_DECISIONS_PER_RACE
        return Step(
            observation=observation,
            reward=reward,
            finished=finished,
            out_of_time=not finished and self._decisions >= limit,
            progress=progress,
            decisions=self._decisions,
            pick_ignored=pick_ignored,
            pedal_ignored=pedal_ignored,
            steering_ignored=steering_ignored,
            boost_started=boost_started,
            race_seconds=self._game.kart.finish_time() if finished else self._game.kart.race_clock(),
            from_archive=self._from_archive,
            held_action=action,
        )

    def _observe(self, buttons: Button) -> np.ndarray:
        return observe(self._game.kart, self._game.track, self._sensors, buttons)
