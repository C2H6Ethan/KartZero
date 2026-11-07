"""Tests for splitting a recording into races, and for the network's outputs."""

import pytest
import torch

from kartzero.environment import ACTIONS, OBSERVATION_SIZE
from kartzero.learner import chunk_advantages, chunk_points, policy_score
from kartzero.network import DriverNetwork


def test_a_chunk_with_no_race_end_uses_the_guess_at_the_end():
    assert chunk_points([1, 1], [False, False], [99, 4], gamma=0.5) == pytest.approx([2.5, 3])


def test_points_from_a_new_race_do_not_leak_into_the_finished_one():
    # The race ends at the first decision, so the 100 after it must not count.
    assert chunk_points([1, 100], [True, False], [0, 0], gamma=0.5) == pytest.approx([1, 100])


def test_a_race_that_ran_out_of_time_still_counts_its_guess():
    assert chunk_points([1, 5], [True, False], [10, 0], gamma=0.5) == pytest.approx([6, 5])


def test_every_decision_gets_points():
    assert len(chunk_points([1, 2, 3, 4], [False, True, False, False], [0, 0, 0, 0], gamma=0.5)) == 4


def test_ignored_picks_do_not_change_the_score():
    ratios = torch.tensor([1.1, 0.9, 1.0, 5.0])
    advantages = torch.tensor([2.0, -1.0, 0.5, 100.0])
    used = torch.tensor([True, True, True, False])
    assert policy_score(ratios, advantages, used) == pytest.approx(
        policy_score(ratios[:3], advantages[:3], torch.ones(3, dtype=torch.bool)).item()
    )


def test_ignored_picks_get_no_push():
    ratios = torch.ones(3, requires_grad=True)
    policy_score(ratios, torch.tensor([1.0, -1.0, 7.0]), torch.tensor([True, True, False])).backward()
    assert ratios.grad is not None and ratios.grad[2] == 0


def test_a_batch_with_too_few_used_picks_scores_zero():
    score = policy_score(torch.ones(3), torch.tensor([1.0, 2.0, 3.0]), torch.tensor([False, True, False]))
    assert score.item() == 0.0


def test_the_network_gives_odds_for_every_action_and_one_guess_per_row():
    odds, guesses = DriverNetwork(OBSERVATION_SIZE, len(ACTIONS))(torch.zeros(3, OBSERVATION_SIZE))
    assert odds.probs.shape == (3, len(ACTIONS))
    assert guesses.shape == (3,)


def test_a_new_network_starts_with_about_equal_odds():
    odds, _ = DriverNetwork(OBSERVATION_SIZE, len(ACTIONS))(torch.zeros(1, OBSERVATION_SIZE))
    assert odds.probs[0].tolist() == pytest.approx([1 / len(ACTIONS)] * len(ACTIONS), abs=0.02)


def test_chunk_advantages_with_a_blend_of_one_match_the_real_points():
    rewards, ends, after, guesses = [1, 100, 2], [True, False, False], [0, 0, 7], [3, 50, 1]
    real = chunk_points(rewards, ends, after, gamma=0.5)
    expected = [p - g for p, g in zip(real, guesses)]
    assert chunk_advantages(rewards, guesses, ends, after, gamma=0.5, blend=1.0) == pytest.approx(expected)


def test_chunk_advantages_do_not_leak_between_races():
    # The race ends at the first decision, so the big second race can't change the first one.
    first = chunk_advantages([1, 100], [0, 0], [True, False], [0, 0], gamma=0.5, blend=0.9)[0]
    assert first == pytest.approx(1)


def test_used_log_odds_pick_the_half_that_was_used():
    from kartzero.learner import used_log_odds

    logits = torch.randn(4, 9)
    odds = torch.distributions.Categorical(logits=logits)
    probs = odds.probs.view(4, 3, 3)
    actions = torch.tensor([7, 7, 7, 5])  # pedal 2 steering 1, and pedal 1 steering 2
    pedal_used = torch.tensor([True, True, False, False])
    steering_used = torch.tensor([True, False, True, False])
    result = used_log_odds(odds, actions, pedal_used, steering_used)
    assert result[0] == pytest.approx(odds.log_prob(actions)[0].item(), abs=1e-5)
    assert result[1] == pytest.approx(probs[1, 2, :].sum().log().item(), abs=1e-5)
    assert result[2] == pytest.approx(probs[2, :, 1].sum().log().item(), abs=1e-5)
    assert result[3] == pytest.approx(odds.log_prob(actions)[3].item(), abs=1e-5)
