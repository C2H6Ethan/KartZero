"""What the AI's inputs must be, using a fake kart and track so that no game file is needed."""

import math

import numpy as np
import pytest

from kartzero.game import Button
from kartzero.observation import observe, view_of_point

EAST = 0.0
NORTH = math.pi / 2  # angles go from the x axis towards +z
AT_ORIGIN = (0.0, 50.0, 0.0)


def test_a_point_straight_ahead_has_angle_zero():
    assert view_of_point(AT_ORIGIN, EAST, (100.0, 0.0)) == pytest.approx((100.0, 0.0))


def test_distance_ignores_height_and_uses_both_directions():
    assert view_of_point(AT_ORIGIN, EAST, (30.0, 40.0))[0] == pytest.approx(50.0)


def test_a_point_on_one_side_is_positive_and_the_other_negative():
    assert view_of_point(AT_ORIGIN, EAST, (0.0, 100.0))[1] == pytest.approx(math.pi / 2)
    assert view_of_point(AT_ORIGIN, EAST, (0.0, -100.0))[1] == pytest.approx(-math.pi / 2)


def test_the_angle_depends_on_where_the_kart_points():
    assert view_of_point(AT_ORIGIN, NORTH, (0.0, 100.0))[1] == pytest.approx(0.0)
    assert view_of_point(AT_ORIGIN, NORTH, (100.0, 0.0))[1] == pytest.approx(-math.pi / 2)


def test_the_kart_does_not_have_to_be_at_the_origin():
    assert view_of_point((10.0, 0.0, 20.0), EAST, (110.0, 20.0)) == pytest.approx((100.0, 0.0))


@pytest.mark.parametrize("heading, point, expected", [
    (math.radians(170), (math.cos(math.radians(-170)) * 100, math.sin(math.radians(-170)) * 100), 20),
    (math.radians(-170), (math.cos(math.radians(170)) * 100, math.sin(math.radians(170)) * 100), -20),
])
def test_the_angle_takes_the_short_way_round(heading, point, expected):
    assert math.degrees(view_of_point(AT_ORIGIN, heading, point)[1]) == pytest.approx(expected)


class FakeKart:
    def speed(self):
        return 5.0

    def drift_angle(self):
        return -30.0

    def turning_rate(self):
        return 150

    def mini_turbo_count(self):
        return 2

    def boost_time_left(self):
        return 11

    def is_off_road(self):
        return True

    def is_on_ground(self):
        return False

    def position(self):
        return AT_ORIGIN

    def heading(self):
        return EAST

    def checkpoint(self):
        return 24  # with 26 checkpoints, the ones ahead are 25, then 0, then 1


class FakeTrack:
    CENTERS = {25: (100.0, 0.0), 0: (0.0, 200.0), 1: (0.0, -400.0)}

    def checkpoint_count(self):
        return 26

    def checkpoint_center(self, index):
        return self.CENTERS[index]


class FakeSensors:
    def wall_distances(self, position, heading):
        return np.array([1000.0, 500.0])

    def road_edge_distances(self, position, heading):
        return np.array([250.0, 100.0])


@pytest.fixture
def numbers():
    return observe(FakeKart(), FakeTrack(), FakeSensors(), Button.A | Button.R)


def test_the_type_is_what_the_network_needs(numbers):
    assert numbers.dtype == np.float32


def test_kart_state_is_scaled_in_order(numbers):
    assert numbers[:5] == pytest.approx([0.5, -0.5, 0.5, 0.5, 0.2])


def test_true_and_false_become_one_and_zero(numbers):
    assert numbers[5:7].tolist() == [1.0, 0.0]


def test_held_buttons_come_next(numbers):
    # A, R, LEFT, RIGHT, with A and R held.
    assert numbers[7:11].tolist() == [1.0, 1.0, 0.0, 0.0]


def test_distances_are_scaled(numbers):
    assert numbers[11:15] == pytest.approx([1.0, 0.5, 0.25, 0.1])


def test_checkpoints_ahead_wrap_around_the_end_of_the_lap(numbers):
    assert numbers[15:] == pytest.approx([0.1, 0.0, 0.2, 0.5, 0.4, -0.5])


def test_nothing_extra_is_included(numbers):
    assert len(numbers) == 21
