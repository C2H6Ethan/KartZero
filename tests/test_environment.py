"""Tests for one race, using a fake game so that no game file is needed."""

import numpy as np
import pytest

from kartzero.environment import ACTIONS, MAX_DECISIONS_PER_RACE, OBSERVATION_SIZE, RaceEnvironment
from kartzero.game import Button


class FakeKart:
    def __init__(self):
        self.progress = 0.0
        self.snapshot_loaded = False

    def race_progress(self):
        return self.progress

    def has_finished(self):
        return self.progress >= 1.0

    def speed(self):
        return 1.0 if self.snapshot_loaded else 99.0

    def drift_angle(self):
        return 0.0

    def turning_rate(self):
        return 0

    def mini_turbo_count(self):
        return 0

    def boost_time_left(self):
        return 0

    def is_off_road(self):
        return False

    def is_on_ground(self):
        return True

    def position(self):
        return (0.0, 0.0, 0.0)

    def heading(self):
        return 0.0

    def checkpoint(self):
        return 0


class FakeTrack:
    def checkpoint_count(self):
        return 26

    def checkpoint_center(self, index):
        return (100.0, 0.0)


class FakeSensors:
    def wall_distances(self, position, heading):
        return np.zeros(9)

    def road_edge_distances(self, position, heading):
        return np.zeros(9)


class FakeGame:
    def __init__(self, progress_per_decision=0.0):
        self.kart = FakeKart()
        self.track = FakeTrack()
        self.held = []
        self.progress_per_decision = progress_per_decision

    def start_race(self, buttons):
        self.kart.snapshot_loaded = True
        self.kart.progress = 0.0

    def hold(self, buttons):
        self.held.append(buttons)

    def advance(self, frames):
        self.kart.progress += self.progress_per_decision


def make_environment(progress_per_decision=0.0):
    game = FakeGame(progress_per_decision)
    return game, RaceEnvironment(game, FakeSensors())


def test_the_first_numbers_come_from_after_the_snapshot_loads():
    _, environment = make_environment()
    observation = environment.reset()
    assert len(observation) == OBSERVATION_SIZE
    assert observation[0] == pytest.approx(0.1)  # speed 1 / 10, not the 99 from before loading


def test_no_action_includes_the_item_button():
    assert not any(action & Button.L for action in ACTIONS)


def test_no_action_includes_reverse():
    assert not any(action & Button.B for action in ACTIONS)


def test_a_step_holds_the_chosen_action():
    game, environment = make_environment()
    environment.reset()
    environment.step(4)
    assert game.held[-1] == ACTIONS[4]


def test_a_step_scores_only_that_decisions_progress():
    _, environment = make_environment(progress_per_decision=0.001)
    environment.reset()
    environment.step(3)
    assert environment.step(3).reward == pytest.approx(2000 * 0.001 - 0.25)


def test_the_race_ends_as_finished_at_the_line():
    _, environment = make_environment(progress_per_decision=0.6)
    environment.reset()
    assert not environment.step(3).finished
    assert environment.step(3).finished


def test_the_race_ends_as_out_of_time_at_the_limit():
    _, environment = make_environment()
    environment.reset()
    for _ in range(MAX_DECISIONS_PER_RACE - 1):
        assert not environment.step(0).out_of_time
    step = environment.step(0)
    assert step.out_of_time and not step.finished


def test_a_new_race_starts_clean():
    _, environment = make_environment(progress_per_decision=0.6)
    environment.reset()
    environment.step(3)
    environment.step(3)
    environment.reset()
    step = environment.step(3)
    assert step.decisions == 1 and not step.finished
