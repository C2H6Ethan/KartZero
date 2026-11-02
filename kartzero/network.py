"""The AI's brain: from the 36 numbers to button odds and a guess of future points."""

import torch
from torch import nn

HIDDEN_SIZE = 256
VALUE_SCALE = 100.0  # future points reach the hundreds, so the network outputs them in hundreds


def _layers(inputs: int, outputs: int, last_layer_size: float) -> nn.Sequential:
    last = nn.Linear(HIDDEN_SIZE, outputs)
    nn.init.orthogonal_(last.weight, last_layer_size)
    nn.init.zeros_(last.bias)
    return nn.Sequential(
        nn.Linear(inputs, HIDDEN_SIZE), nn.Tanh(), nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.Tanh(), last
    )


class DriverNetwork(nn.Module):
    """Two small networks side by side, one for the odds and one for the guess."""

    def __init__(self, inputs: int, actions: int) -> None:
        super().__init__()
        self._odds = _layers(inputs, actions, last_layer_size=0.01)  # small, so all actions start about equal
        self._guess = _layers(inputs, 1, last_layer_size=1.0)

    def forward(self, observations: torch.Tensor) -> tuple[torch.distributions.Categorical, torch.Tensor]:
        """The odds of each action, and the guess of future points, for each row of numbers."""
        return (
            torch.distributions.Categorical(logits=self._odds(observations)),
            self._guess(observations).squeeze(-1) * VALUE_SCALE,
        )
