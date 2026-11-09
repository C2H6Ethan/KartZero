"""The AI's brain: from the numbers it receives to button odds and a guess of future points."""

from pathlib import Path

import torch
from torch import nn

from kartzero.environment import COUNTDOWN_INPUT, OBSERVATION_SIZE

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
        self.inputs = inputs
        self.actions = actions
        self._odds = _layers(inputs, actions, last_layer_size=0.01)  # small, so all actions start about equal
        self._guess = _layers(inputs, 1, last_layer_size=1.0)

    def forward(self, observations: torch.Tensor) -> tuple[torch.distributions.Categorical, torch.Tensor]:
        """The odds of each action, and the guess of future points, for each row of numbers."""
        return (
            torch.distributions.Categorical(logits=self._odds(observations)),
            self._guess(observations).squeeze(-1) * VALUE_SCALE,
        )


# Brains saved before the countdown input was added: 35 inputs, or 37 with hold lengths.
INPUTS_WITHOUT_COUNTDOWN = (OBSERVATION_SIZE - 1, OBSERVATION_SIZE + 1)


def network_state(file: Path) -> dict[str, torch.Tensor]:
    """A saved brain's settings. A brain from before the countdown input gets that input added with
    weights of 0, so it drives exactly as it did until training teaches it to use the countdown."""
    state: dict[str, torch.Tensor] = torch.load(file, map_location="cpu")["network"]
    if state["_odds.0.weight"].shape[1] in INPUTS_WITHOUT_COUNTDOWN:
        for key in ("_odds.0.weight", "_guess.0.weight"):
            weights = state[key]
            no_weight = torch.zeros(weights.shape[0], 1, dtype=weights.dtype)
            state[key] = torch.cat([weights[:, :COUNTDOWN_INPUT], no_weight, weights[:, COUNTDOWN_INPUT:]], dim=1)
    return state


def load_driver(file: Path) -> DriverNetwork:
    """Load a saved brain, whatever its size: the sizes of its first and last layers say what it is."""
    state = network_state(file)
    first_layer, last_layer = state["_odds.0.weight"], state["_odds.4.weight"]
    network = DriverNetwork(inputs=first_layer.shape[1], actions=last_layer.shape[0])
    network.load_state_dict(state)
    return network
