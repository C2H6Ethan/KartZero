"""Curiosity by Random Network Distillation (Burda et al. 2018).

A fixed random network turns each set of inputs into some numbers. A second network learns to
predict those numbers. On inputs it has seen a lot, its prediction gets good. On new ones it's
still bad. So its error says how new a moment is, and that can be paid as a small bonus, which
draws the AI towards moments it hasn't tried yet. The bonus fades by itself as moments get familiar.

It knows nothing about drifting or this game: every input counts the same, so it's allowed as
self-taught exploration (decided 7 November). Test laps still measure only the race time.
"""

import torch
from torch import nn

FEATURES = 64
HIDDEN = 128
LEARNING_RATE = 1e-4
EPSILON = 1e-8
MAX_NOVELTY = 5.0  # so one odd moment can't swamp everything else


def _network(inputs: int, hidden_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    size = inputs
    for _ in range(hidden_layers):
        layers += [nn.Linear(size, HIDDEN), nn.ReLU()]
        size = HIDDEN
    layers.append(nn.Linear(size, FEATURES))
    return nn.Sequential(*layers)


class Curiosity:
    """Scores how new each moment is, and learns from the moments it's shown."""

    def __init__(self, inputs: int, device: torch.device, seed: int = 0) -> None:
        generator_state = torch.random.get_rng_state()
        torch.manual_seed(seed + 12345)  # its own random networks, without disturbing the AI's seed
        self._target = _network(inputs, 1).to(device)
        self._predictor = _network(inputs, 2).to(device)
        torch.random.set_rng_state(generator_state)
        for parameter in self._target.parameters():
            parameter.requires_grad_(False)
        self._optimizer = torch.optim.Adam(self._predictor.parameters(), lr=LEARNING_RATE)
        self._mean = torch.zeros(inputs, device=device)
        self._variance = torch.ones(inputs, device=device)
        self._seen = 0
        self._novelty_scale = 1.0

    def bonus(self, observations: torch.Tensor, weight: float) -> torch.Tensor:
        """The bonus for reaching each moment: above 0 for newer than average, below 0 for more familiar.

        Centred on 0 within the batch, so on average it neither rewards nor punishes longer races.
        """
        novelty = self.novelty(observations).clamp(max=MAX_NOVELTY)
        return weight * (novelty - novelty.mean())

    def novelty(self, observations: torch.Tensor) -> torch.Tensor:
        """How new each moment is, scaled so that a typical moment of the last chunk scores about 1."""
        with torch.no_grad():
            error = self._error(observations)
        return error / self._novelty_scale

    def learn(self, observations: torch.Tensor, passes: int = 4, batch_size: int = 512) -> None:
        """Update the input averages and the predictor, so these moments count as less new next time."""
        self._update_averages(observations)
        with torch.no_grad():
            self._novelty_scale = float(self._error(observations).mean()) + EPSILON
        for _ in range(passes):
            order = torch.randperm(len(observations), device=observations.device)
            for start in range(0, len(observations), batch_size):
                loss = self._error(observations[order[start : start + batch_size]]).mean()
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

    def _error(self, observations: torch.Tensor) -> torch.Tensor:
        normalised = ((observations - self._mean) / (self._variance.sqrt() + EPSILON)).clamp(-5, 5)
        return ((self._predictor(normalised) - self._target(normalised)) ** 2).mean(dim=1)

    def _update_averages(self, observations: torch.Tensor) -> None:
        count = len(observations)
        batch_mean = observations.mean(dim=0)
        batch_variance = observations.var(dim=0, unbiased=False)
        total = self._seen + count
        difference = batch_mean - self._mean
        self._mean = self._mean + difference * count / total
        self._variance = (
            self._variance * self._seen + batch_variance * count + difference**2 * self._seen * count / total
        ) / total
        self._seen = total
