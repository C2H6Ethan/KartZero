"""Tests for one race, using a fake game so that no game file is needed."""

import numpy as np
import pytest

from kartzero.environment import (
    ACTIONS,
    MAX_DECISIONS_PER_RACE,
    OBSERVATION_SIZE,
    PEDALS,
    STEERINGS,
    RaceEnvironment,
)
from kartzero.game import Button


class FakeKart:
    def __init__(self):
        self.progress = 0.0
        self.snapshot_loaded = False
        self.boost = 0
        self.clock = 2.4  # the game's clock starts in the countdown, before the kart can move
        self.official = 0.0

    def race_clock(self):
        return self.clock

    def finish_time(self):
        return self.official

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
        return self.boost

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


def make_environment(progress_per_decision=0.0, sticky_chance=0.0, seed=0):
    game = FakeGame(progress_per_decision)
    return game, RaceEnvironment(game, FakeSensors(), sticky_chance, seed)


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


def test_without_stickiness_every_pick_is_held():
    game, environment = make_environment(sticky_chance=0.0)
    environment.reset()
    for action in [4, 5, 3, 7, 8]:
        assert not environment.step(action).pick_ignored
    assert game.held == [ACTIONS[a] for a in [4, 5, 3, 7, 8]]


def test_full_stickiness_always_repeats_the_last_button():
    game, environment = make_environment(sticky_chance=1.0)
    environment.reset()
    for action in [4, 5, 3]:
        assert environment.step(action).pick_ignored
    assert game.held == [ACTIONS[0]] * 3  # a race starts with nothing held


def test_a_repeat_is_marked_even_when_the_pick_was_the_same_button():
    # Leaving these out only when the pick differed would push the AI towards repeating.
    _, environment = make_environment(sticky_chance=1.0)
    environment.reset()
    assert environment.step(0).pick_ignored


def test_a_new_race_forgets_the_last_button():
    game, environment = make_environment(sticky_chance=0.0)
    environment.reset()
    environment.step(4)
    environment.sticky_chance = 1.0
    environment.reset()
    environment.step(5)
    assert game.held[-1] == ACTIONS[0]


def test_some_stickiness_repeats_about_that_often():
    _, environment = make_environment(sticky_chance=0.25, seed=1)
    environment.reset()
    ignored = sum(environment.step(i % len(ACTIONS)).pick_ignored for i in range(2000))
    assert 400 < ignored < 600


def test_the_same_seed_repeats_the_same_buttons():
    runs = []
    for _ in range(2):
        _, environment = make_environment(sticky_chance=0.5, seed=7)
        environment.reset()
        runs.append([environment.step(i % len(ACTIONS)).pick_ignored for i in range(100)])
    assert runs[0] == runs[1]


def test_a_boost_is_counted_when_its_timer_jumps_up():
    game, environment = make_environment()
    environment.reset()
    assert not environment.step(3).boost_started
    game.kart.boost = 52
    assert environment.step(3).boost_started
    game.kart.boost = 48  # counting down is the same boost
    assert not environment.step(3).boost_started


def test_a_step_reports_the_games_clock_during_the_race():
    game, environment = make_environment()
    environment.reset()
    game.kart.clock = 50.0
    assert environment.step(3).race_seconds == 50.0


def test_a_finish_reports_the_official_time_not_the_clock():
    # The clock keeps running past the line, but the official time is when the kart crossed it.
    game, environment = make_environment(progress_per_decision=1.0)
    environment.reset()
    game.kart.clock, game.kart.official = 103.011, 102.967
    assert environment.step(3).race_seconds == 102.967


def test_a_new_race_starts_clean():
    _, environment = make_environment(progress_per_decision=0.6)
    environment.reset()
    environment.step(3)
    environment.step(3)
    environment.reset()
    step = environment.step(3)
    assert step.decisions == 1 and not step.finished


def test_every_action_is_one_pedal_and_one_steering():
    assert len(ACTIONS) == len(PEDALS) * len(STEERINGS)
    for pedal_index, pedal in enumerate(PEDALS):
        for steering_index, steering in enumerate(STEERINGS):
            assert ACTIONS[pedal_index * len(STEERINGS) + steering_index] == pedal | steering


def test_a_sticky_repeat_ignores_both_halves_of_the_pick():
    _, environment = make_environment(sticky_chance=0.5, seed=3)
    environment.reset()
    for i in range(200):
        step = environment.step(i % len(ACTIONS))
        assert step.pedal_ignored == step.steering_ignored == step.pick_ignored
