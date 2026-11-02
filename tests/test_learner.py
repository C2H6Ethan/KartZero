"""Tests for splitting a recording into races, and for the network's outputs."""

import pytest
import torch

from kartzero.environment import ACTIONS, OBSERVATION_SIZE
from kartzero.learner import chunk_points
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


def test_the_network_gives_odds_for_every_action_and_one_guess_per_row():
    odds, guesses = DriverNetwork(OBSERVATION_SIZE, len(ACTIONS))(torch.zeros(3, OBSERVATION_SIZE))
    assert odds.probs.shape == (3, len(ACTIONS))
    assert guesses.shape == (3,)


def test_a_new_network_starts_with_about_equal_odds():
    odds, _ = DriverNetwork(OBSERVATION_SIZE, len(ACTIONS))(torch.zeros(1, OBSERVATION_SIZE))
    assert odds.probs[0].tolist() == pytest.approx([1 / len(ACTIONS)] * len(ACTIONS), abs=0.02)
