"""Tests for choosing hold lengths, using the fake game from the environment tests."""

from test_environment import FakeGame, FakeSensors

from kartzero.environment import ACTIONS, RaceEnvironment
from kartzero.game import Button
from kartzero.holds import HALF_CHOICES, HOLD_ACTIONS, HOLD_LENGTHS, HOLD_OBSERVATION_SIZE, HoldEnvironment, split

A_R, A, STRAIGHT, LEFT, RIGHT = 2, 1, 0, 1, 2


def action(pedal, pedal_length, steering, steering_length):
    pedal_half = pedal * len(HOLD_LENGTHS) + HOLD_LENGTHS.index(pedal_length)
    steering_half = steering * len(HOLD_LENGTHS) + HOLD_LENGTHS.index(steering_length)
    return pedal_half * HALF_CHOICES + steering_half


def make():
    game = FakeGame()
    return game, HoldEnvironment(RaceEnvironment(game, FakeSensors()))


def test_every_action_number_splits_into_a_pick_and_a_length_for_each_half():
    seen = {split(a) for a in range(HOLD_ACTIONS)}
    assert len(seen) == HOLD_ACTIONS
    assert split(action(A_R, 16, LEFT, 2)) == (A_R, 16, LEFT, 2)


def test_the_pedal_is_held_while_the_steering_changes():
    game, environment = make()
    assert len(environment.reset()) == HOLD_OBSERVATION_SIZE
    environment.step(action(A_R, 4, LEFT, 1))
    for steering in (RIGHT, LEFT, RIGHT):
        environment.step(action(A, 1, steering, 1))  # the pedal pick is ignored while A + R is held
    environment.step(action(A, 1, STRAIGHT, 1))  # the A + R hold has run out
    assert game.held == [
        Button.A | Button.R | Button.LEFT,
        Button.A | Button.R | Button.RIGHT,
        Button.A | Button.R | Button.LEFT,
        Button.A | Button.R | Button.RIGHT,
        Button.A,
    ]


def test_a_step_says_which_halves_took_a_new_pick():
    _, environment = make()
    environment.reset()
    first = environment.step(action(A, 2, LEFT, 1))
    second = environment.step(action(A, 1, RIGHT, 2))
    third = environment.step(action(A, 1, RIGHT, 1))
    assert (first.pedal_ignored, first.steering_ignored) == (False, False)
    assert (second.pedal_ignored, second.steering_ignored, second.pick_ignored) == (True, False, False)
    assert (third.pedal_ignored, third.steering_ignored, third.pick_ignored) == (False, True, False)


def test_the_numbers_show_how_long_each_hold_has_left():
    _, environment = make()
    environment.reset()
    observation = environment.step(action(A, 32, LEFT, 4)).observation
    assert observation[-2] * 32 == 31 and observation[-1] * 32 == 3


def test_a_new_race_drops_any_hold():
    game, environment = make()
    environment.reset()
    environment.step(action(A_R, 32, LEFT, 32))
    environment.reset()
    environment.step(action(A, 1, RIGHT, 1))
    assert game.held[-1] == ACTIONS[1 * 3 + RIGHT]
