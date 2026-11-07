"""Tests for the curiosity bonus."""

import torch

from kartzero.curiosity import Curiosity


def test_moments_seen_often_become_less_new_than_unseen_ones():
    torch.manual_seed(0)
    curiosity = Curiosity(6, torch.device("cpu"))
    familiar = torch.randn(2048, 6) * 0.1
    for _ in range(30):
        curiosity.learn(familiar)
    unseen = torch.randn(256, 6) * 0.1 + torch.tensor([0.0, 0.0, 0.0, 3.0, 0.0, 0.0])
    assert curiosity.novelty(unseen).mean() > 3 * curiosity.novelty(familiar[:256]).mean()


def test_the_bonus_averages_zero_and_scales_with_its_weight():
    curiosity = Curiosity(6, torch.device("cpu"))
    observations = torch.randn(100, 6)
    small, large = curiosity.bonus(observations, 0.1), curiosity.bonus(observations, 0.2)
    assert abs(float(small.mean())) < 1e-6
    assert torch.allclose(large, 2 * small)


def test_curiosity_does_not_change_the_ais_own_random_numbers():
    torch.manual_seed(5)
    expected = torch.rand(3)
    torch.manual_seed(5)
    Curiosity(6, torch.device("cpu"), seed=1)
    assert torch.equal(torch.rand(3), expected)
