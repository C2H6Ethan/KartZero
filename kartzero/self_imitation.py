"""Self-imitation (Oh et al. 2018): learn again from your own past decisions that went better than expected.

PPO uses each decision for one round and then throws it away, so a rare lucky move barely moves
the odds before it's forgotten. This keeps a large buffer of past decisions with the real points
that followed them. Each round, it takes a sample, keeps only the decisions whose points beat the
current guess, and pushes the odds towards the action that was really held, more for bigger
surprises. Decisions that went worse than expected are ignored, so it can only copy good luck.

The reward isn't changed: it's the same points, just learned from more than once.
"""

import numpy as np
import torch

from kartzero.network import VALUE_SCALE, DriverNetwork

GUESS_PULL_WEIGHT = 0.1  # the paper keeps this small, so self-imitation mostly changes the odds


class ExperienceBuffer:
    """The most recent `capacity` decisions: what the AI saw, what it held, and the points that followed."""

    def __init__(self, capacity: int, observation_size: int, seed: int = 0) -> None:
        self._observations = np.zeros((capacity, observation_size), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._points = np.zeros(capacity, dtype=np.float32)
        self._capacity = capacity
        self._next = 0
        self._size = 0
        self._random = np.random.default_rng(seed)

    def add(self, observations: np.ndarray, actions: list[int], points: list[float]) -> None:
        for observation, action, point in zip(observations, actions, points):
            self._observations[self._next] = observation
            self._actions[self._next] = action
            self._points[self._next] = point
            self._next = (self._next + 1) % self._capacity
            self._size = min(self._size + 1, self._capacity)

    def __len__(self) -> int:
        return self._size

    def sample(self, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = self._random.integers(0, self._size, size=min(count, self._size))
        return self._observations[rows], self._actions[rows], self._points[rows]


def self_imitation_loss(
    network: DriverNetwork, observations: torch.Tensor, actions: torch.Tensor, points: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """The loss for one sample, and how many of its decisions beat the guess.

    For each decision that beat the guess by `surprise`: push up the odds of its action by
    surprise, and pull the guess up towards the points. Surprises are in hundreds of points, like the guess.
    """
    odds, guesses = network(observations)
    surprise = ((points - guesses) / VALUE_SCALE).clamp(min=0)
    better = int((surprise > 0).sum())
    if better == 0:
        return (guesses * 0).sum(), 0
    imitation = -(odds.log_prob(actions) * surprise.detach()).sum() / better
    guess_pull = 0.5 * (surprise**2).sum() / better
    return imitation + GUESS_PULL_WEIGHT * guess_pull, better
