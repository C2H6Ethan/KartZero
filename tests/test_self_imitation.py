"""Tests for self-imitation."""

import numpy as np
import torch

from kartzero.network import DriverNetwork
from kartzero.self_imitation import ExperienceBuffer, self_imitation_loss


def test_the_buffer_keeps_only_the_newest_decisions():
    buffer = ExperienceBuffer(capacity=3, observation_size=2)
    buffer.add(np.arange(10, dtype=np.float32).reshape(5, 2), [0, 1, 2, 3, 4], [0.0, 1.0, 2.0, 3.0, 4.0])
    assert len(buffer) == 3
    samples = [buffer.sample(3) for _ in range(30)]
    assert {a for _, actions, _ in samples for a in actions.tolist()} == {2, 3, 4}
    assert {p for _, _, points in samples for p in points.tolist()} == {2.0, 3.0, 4.0}


def test_decisions_that_did_worse_than_the_guess_teach_nothing():
    torch.manual_seed(0)
    network = DriverNetwork(4, 9)
    observations = torch.randn(8, 4)
    _, guesses = network(observations)
    loss, better = self_imitation_loss(network, observations, torch.zeros(8, dtype=torch.long), guesses.detach() - 10)
    assert better == 0
    loss.backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in network.parameters())


def test_a_decision_that_beat_the_guess_makes_its_action_likelier():
    torch.manual_seed(0)
    network = DriverNetwork(4, 9)
    observation = torch.randn(1, 4)
    action = torch.tensor([7])
    optimizer = torch.optim.SGD(network.parameters(), lr=0.5)
    before = network(observation)[0].probs[0, 7].item()
    for _ in range(5):
        _, guess = network(observation)
        loss, better = self_imitation_loss(network, observation, action, guess.detach() + 50)
        assert better == 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert network(observation)[0].probs[0, 7].item() > before
