"""Tests for official test laps, using fakes so that no game file is needed."""

import numpy as np
import pytest
import torch

from kartzero.environment import Step
from kartzero.test_lap import drive_test_lap
from kartzero.versions import LapResult, VersionStore


class FakeNetwork:
    """Always prefers the action it's given, with the odds of the others above zero."""

    def __init__(self, favourite, actions=4):
        self.favourite = favourite
        self.actions = actions

    def __call__(self, observations):
        logits = torch.zeros(1, self.actions)
        logits[0, self.favourite] = 1.0
        return torch.distributions.Categorical(logits=logits), torch.zeros(1)


class FakeEnvironment:
    def __init__(self, decisions_to_end, finishes=True):
        self.decisions_to_end = decisions_to_end
        self.finishes = finishes
        self.chosen = []
        self.resets = 0

    def reset(self):
        self.resets += 1
        return np.zeros(3, dtype=np.float32)

    def step(self, action):
        self.chosen.append(action)
        ended = len(self.chosen) >= self.decisions_to_end
        return Step(
            observation=np.zeros(3, dtype=np.float32),
            reward=0.0,
            finished=ended and self.finishes,
            out_of_time=ended and not self.finishes,
            progress=1.0 if ended and self.finishes else 0.5,
            decisions=len(self.chosen),
        )


def test_it_only_ever_picks_the_likeliest_action():
    environment = FakeEnvironment(decisions_to_end=200)
    drive_test_lap(FakeNetwork(favourite=2), environment)
    assert set(environment.chosen) == {2}


def test_it_starts_a_new_race():
    environment = FakeEnvironment(decisions_to_end=1)
    drive_test_lap(FakeNetwork(favourite=0), environment)
    assert environment.resets == 1


def test_a_finished_lap_reports_its_time():
    result = drive_test_lap(FakeNetwork(favourite=0), FakeEnvironment(decisions_to_end=1500))
    assert result == LapResult(True, 100.0, 1.0)


def test_a_lap_that_runs_out_of_time_reports_how_far_it_got():
    result = drive_test_lap(FakeNetwork(favourite=0), FakeEnvironment(decisions_to_end=3, finishes=False))
    assert not result.finished
    assert result.progress == 0.5


@pytest.mark.parametrize("result, text", [
    (LapResult(True, 87.8, 1.0), "1:27.80"),
    (LapResult(True, 109.2, 1.0), "1:49.20"),
    (LapResult(False, 200.0, 0.533), "53.3%"),
])
def test_results_read_like_race_times(result, text):
    assert result.description() == text


def test_saved_laps_come_back_with_their_version(tmp_path):
    store = VersionStore(tmp_path)
    file = store.save({"weight": torch.tensor(1.0)}, 25_000, 1.0, 0.5)
    assert store.test_laps() == {}
    store.save_test_lap(file, LapResult(True, 109.25, 1.0))
    assert store.test_laps()[file.name] == LapResult(True, 109.25, 1.0)
