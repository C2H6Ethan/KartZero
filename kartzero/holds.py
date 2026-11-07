"""Let the AI choose how long to hold each half of its action (an idea from FiGAR, Sharma et al. 2017).

Each decision, the AI picks a pedal (nothing, A, A + R) with a hold length, and a steering (straight,
LEFT, RIGHT) with a hold length. A half only takes a new pick when its last hold has run out, so the
pedal can stay on A + R through several short steering holds. Both halves get the same lengths to
choose from, so no button is favoured. The race and the reward are exactly as before.

An action is one number: pedal half * HALF_CHOICES + steering half, where each half is
value * len(HOLD_LENGTHS) + length index. So there are HALF_CHOICES ** 2 actions.
"""

from dataclasses import replace

import numpy as np

from kartzero.environment import ACTIONS, OBSERVATION_SIZE, STEERING_CHOICES, RaceEnvironment, Step
from kartzero.settings import Settings

HOLD_LENGTHS = (1, 2, 4, 8, 16, 32)  # in decisions
HALF_CHOICES = 3 * len(HOLD_LENGTHS)
HOLD_ACTIONS = HALF_CHOICES**2
HOLD_OBSERVATION_SIZE = OBSERVATION_SIZE + 2  # plus how long each half's hold has left
LONGEST_HOLD = max(HOLD_LENGTHS)


def sizes(settings: Settings) -> tuple[int, int, int]:
    """(inputs, actions, choices per half) for the AI these settings train."""
    if settings.hold_lengths:
        return HOLD_OBSERVATION_SIZE, HOLD_ACTIONS, HALF_CHOICES
    return OBSERVATION_SIZE, len(ACTIONS), STEERING_CHOICES


def make_environment(race: RaceEnvironment, settings: Settings) -> "RaceEnvironment | HoldEnvironment":
    return HoldEnvironment(race) if settings.hold_lengths else race


def split(action: int) -> tuple[int, int, int, int]:
    """(pedal, pedal hold length, steering, steering hold length) for one action number."""
    pedal_half, steering_half = divmod(action, HALF_CHOICES)
    pedal, pedal_length = divmod(pedal_half, len(HOLD_LENGTHS))
    steering, steering_length = divmod(steering_half, len(HOLD_LENGTHS))
    return pedal, HOLD_LENGTHS[pedal_length], steering, HOLD_LENGTHS[steering_length]


class HoldEnvironment:
    """A race where each half of the action is held for a length the AI chose."""

    def __init__(self, race: RaceEnvironment) -> None:
        self._race = race
        self._pedal = self._steering = 0
        self._pedal_left = self._steering_left = 0

    def reset(self) -> np.ndarray:
        self._pedal_left = self._steering_left = 0
        return self._observe(self._race.reset())

    def step(self, action: int) -> Step:
        pedal, pedal_length, steering, steering_length = split(action)
        pedal_decides = self._pedal_left == 0
        steering_decides = self._steering_left == 0
        if pedal_decides:
            self._pedal, self._pedal_left = pedal, pedal_length
        if steering_decides:
            self._steering, self._steering_left = steering, steering_length
        step = self._race.step(self._pedal * STEERING_CHOICES + self._steering)
        self._pedal_left -= 1
        self._steering_left -= 1
        return replace(
            step,
            observation=self._observe(step.observation),
            pedal_ignored=not pedal_decides,
            steering_ignored=not steering_decides,
            pick_ignored=not (pedal_decides or steering_decides),
        )

    def _observe(self, observation: np.ndarray) -> np.ndarray:
        left = np.array([self._pedal_left, self._steering_left], dtype=np.float32) / LONGEST_HOLD
        return np.concatenate([observation, left])
