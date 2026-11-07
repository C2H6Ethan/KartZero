"""Hold lengths: the AI also picks how long to hold each half of its action (from FiGAR, Sharma et al. 2017).

Each decision, the AI picks a pedal (nothing, A, A + R) with a hold length, and a steering (straight,
LEFT, RIGHT) with a hold length. A half only takes a new pick when its last hold has run out. So the
AI can choose to keep A + R held while the steering changes underneath, which is what a mini-turbo
needs, and because it's the AI's own choice, PPO can learn from it. Both halves get the same lengths
to choose from, so no button is favoured. The race and the reward are exactly as before.

An action is one number made of two halves, each a (choice, length) pair:
    action = pedal half * STEERING_HALVES + steering half,  half = choice * len(HOLD_LENGTHS) + length index
"""

from dataclasses import dataclass, replace
from typing import Union

import numpy as np

from kartzero.environment import ACTIONS, OBSERVATION_SIZE, PEDALS, STEERINGS, RaceEnvironment, Step
from kartzero.network import DriverNetwork
from kartzero.settings import Settings

HOLD_LENGTHS = (1, 2, 4, 8, 16, 32)  # in decisions
PEDAL_HALVES = len(PEDALS) * len(HOLD_LENGTHS)
STEERING_HALVES = len(STEERINGS) * len(HOLD_LENGTHS)
HOLD_ACTIONS = PEDAL_HALVES * STEERING_HALVES
HOLD_OBSERVATION_SIZE = OBSERVATION_SIZE + 2  # plus how long each half's hold has left


@dataclass(frozen=True)
class Layout:
    """The shape of the AI's choices: how many inputs and actions, and how an action splits in two."""

    inputs: int
    actions: int
    steering_half: int  # an action is pedal half * steering_half + steering half


PLAIN = Layout(OBSERVATION_SIZE, len(ACTIONS), len(STEERINGS))
WITH_HOLDS = Layout(HOLD_OBSERVATION_SIZE, HOLD_ACTIONS, STEERING_HALVES)


def layout_for(settings: Settings) -> Layout:
    return WITH_HOLDS if settings.hold_lengths else PLAIN


def race_for(driver: DriverNetwork, race: RaceEnvironment) -> Union[RaceEnvironment, "HoldEnvironment"]:
    """The race this brain was trained to drive: with hold lengths if it has their number of actions."""
    return HoldEnvironment(race) if driver.actions == HOLD_ACTIONS else race


def split(action: int) -> tuple[int, int, int, int]:
    """(pedal, pedal hold length, steering, steering hold length) for one action number."""
    pedal_half, steering_half = divmod(action, STEERING_HALVES)
    pedal, pedal_length = divmod(pedal_half, len(HOLD_LENGTHS))
    steering, steering_length = divmod(steering_half, len(HOLD_LENGTHS))
    return pedal, HOLD_LENGTHS[pedal_length], steering, HOLD_LENGTHS[steering_length]


class HoldEnvironment:
    """A race where each half of the action is held for the length the AI chose."""

    def __init__(self, race: RaceEnvironment) -> None:
        self._race = race
        self._pedal = self._steering = 0
        self._pedal_left = self._steering_left = 0  # decisions until that half picks again

    def reset(self) -> np.ndarray:
        self._pedal_left = self._steering_left = 0
        return self._with_holds_left(self._race.reset())

    def step(self, action: int) -> Step:
        pedal, pedal_length, steering, steering_length = split(action)
        pedal_picks = self._pedal_left == 0
        steering_picks = self._steering_left == 0
        if pedal_picks:
            self._pedal, self._pedal_left = pedal, pedal_length
        if steering_picks:
            self._steering, self._steering_left = steering, steering_length
        step = self._race.step(self._pedal * len(STEERINGS) + self._steering)
        self._pedal_left -= 1
        self._steering_left -= 1
        return replace(
            step,
            observation=self._with_holds_left(step.observation),
            pedal_ignored=not pedal_picks,
            steering_ignored=not steering_picks,
        )

    def _with_holds_left(self, observation: np.ndarray) -> np.ndarray:
        """Add how long each hold has left, from 0 to just under 1, so the AI knows what's coming."""
        holds_left = np.array([self._pedal_left, self._steering_left], dtype=np.float32) / max(HOLD_LENGTHS)
        return np.concatenate([observation, holds_left])
