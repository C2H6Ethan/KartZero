"""The maths of learning from a recorded run."""

import torch

SPEED_LIMIT = 0.2  # how far the odds may move from the old odds before the push stops


def future_points(rewards: list[float], gamma: float, points_after: float = 0.0) -> list[float]:
    """For each decision: its own reward, plus gamma times the future points of the next decision.

    `points_after` is what the last decision's next decision would be worth. Use 0 when the
    race really ended. When a recording stops partway through a race, use the value guess there.
    """
    res = []
    total = points_after
    for r in reversed(rewards):
        total = r + gamma * total
        res.append(total)
    
    return res[::-1]


def advantages(points: list[float], guesses: list[float]) -> list[float]:
    """For each decision: how much better than expected it went, as the real points minus the guess.

    Above 0 means better than expected, below 0 means worse, and 0 means as expected.
    The two lists must be the same length, because they describe the same decisions.
    """
    if len(points) != len(guesses):
        raise ValueError("points and guesses must have the same length")
    return [p - g for p, g in zip(points, guesses)]


def limited_scores(ratios: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
    """For each decision: the score training tries to raise, with PPO's speed limit applied.

    `ratios` is new odds / old odds for the button that was picked. For each decision, work out
    ratio * advantage, then the same with the ratio held between 1 - SPEED_LIMIT and
    1 + SPEED_LIMIT, and return the smaller of the two.
    """
    
    without_clamp = ratios * advantages
    limited_ratios = torch.clamp(ratios, 1 - SPEED_LIMIT, 1 + SPEED_LIMIT)
    with_clamp = limited_ratios * advantages
    return torch.min(without_clamp, with_clamp)


def guess_error(guesses: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """How wrong the value guesses were: the average of (guess - real points) squared.

    Returns a single number. Training lowers it, which moves the guesses towards the real points.
    """
    return ((guesses - points) ** 2).mean()

