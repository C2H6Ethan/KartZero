"""Tests for the distance sensors, using a tiny made-up track so that no game file is needed."""

import math

import numpy as np
import pytest

from kartzero.sensors import MAX_DISTANCE, Lines, RoadGrid, TrackSensors, wall_lines
from kartzero.track import CollisionTriangles

FACING_EAST = np.array([0.0])  # angles are measured from the x axis, towards +z


def wall_across_at(x, lowest=-50, highest=50):
    return Lines([[x, -100]], [[x, 100]], [lowest], [highest])


def test_ray_measures_the_distance_to_a_line_ahead():
    assert wall_across_at(10).distances(0, 0, 0, FACING_EAST)[0] == pytest.approx(10)


def test_a_line_behind_the_kart_is_not_seen():
    assert wall_across_at(-10).distances(0, 0, 0, FACING_EAST)[0] == MAX_DISTANCE


def test_a_line_off_to_the_side_is_missed():
    line = Lines([[10, 50]], [[10, 100]], [-50], [50])
    assert line.distances(0, 0, 0, FACING_EAST)[0] == MAX_DISTANCE


def test_the_nearest_of_two_lines_counts():
    lines = Lines([[20, -100], [10, -100]], [[20, 100], [10, 100]], [-50, -50], [50, 50])
    assert lines.distances(0, 0, 0, FACING_EAST)[0] == pytest.approx(10)


def test_a_slanted_ray_travels_further():
    angle = np.array([math.radians(60)])
    assert wall_across_at(10).distances(0, 0, 0, angle)[0] == pytest.approx(20)


def test_a_line_above_or_below_the_kart_does_not_block_it():
    assert wall_across_at(10, lowest=100, highest=150).distances(0, 0, 0, FACING_EAST)[0] == MAX_DISTANCE


def test_no_lines_at_all_reads_as_far_away():
    assert Lines([], [], [], []).distances(0, 0, 0, FACING_EAST)[0] == MAX_DISTANCE


def triangles(corners, is_wall, floor_type):
    corners = np.array(corners, dtype=float)
    walls = np.array(is_wall)
    return CollisionTriangles(corners, walls, ~walls, np.array(floor_type))


def test_wall_triangle_becomes_a_line_seen_from_above():
    upright = [[[10, 0, -5], [10, 0, 5], [10, 40, 5]]]
    lines = wall_lines(triangles(upright, [True], [8]))
    assert lines.distances(0, 0, 20, FACING_EAST)[0] == pytest.approx(10)
    assert lines.distances(0, 0, 90, FACING_EAST)[0] == MAX_DISTANCE


SQUARE_ROAD = [
    [[0, 0, 0], [20, 0, 0], [20, 0, 20]],
    [[0, 0, 0], [20, 0, 20], [0, 0, 20]],
]


GRASS = [[[20, 0, 0], [40, 0, 0], [40, 0, 20]], [[20, 0, 0], [40, 0, 20], [20, 0, 20]]]
GRID_ACCURACY = 8  # two grid squares: painting includes border squares, and rays move in steps


def test_the_join_between_road_triangles_is_not_an_edge():
    # The diagonal join is 5 away going east from (5, 10), so 15 proves it's ignored.
    grid = RoadGrid(triangles(SQUARE_ROAD, [False, False], [0, 0]))
    assert grid.distances(5, 10, FACING_EAST)[0] == pytest.approx(15, abs=GRID_ACCURACY)


def test_off_road_triangles_do_not_count_as_road():
    grid = RoadGrid(triangles(SQUARE_ROAD + GRASS, [False] * 4, [0, 0, 3, 3]))
    assert grid.distances(5, 10, FACING_EAST)[0] == pytest.approx(15, abs=GRID_ACCURACY)
    assert not grid.is_road(np.array(30.0), np.array(10.0))


def test_off_the_road_it_measures_the_way_back_to_the_road():
    grid = RoadGrid(triangles(SQUARE_ROAD + GRASS, [False] * 4, [0, 0, 3, 3]))
    assert grid.distances(30, 10, np.array([math.pi]))[0] == pytest.approx(10, abs=GRID_ACCURACY)


def test_no_change_in_range_reads_as_far_away():
    grid = RoadGrid(triangles(SQUARE_ROAD, [False, False], [0, 0]))
    assert grid.distances(500, 500, np.array([0.0]))[0] == MAX_DISTANCE


def test_sensors_turn_with_the_kart():
    upright = [[[10, 0, -5], [10, 0, 5], [10, 40, 5]]]
    sensors = TrackSensors(triangles(SQUARE_ROAD + upright, [False, False, True], [0, 0, 8]))
    ahead = sensors.wall_distances((0, 0, 0), heading=0.0)
    turned_away = sensors.wall_distances((0, 0, 0), heading=math.pi)
    assert ahead.min() == pytest.approx(10)
    assert turned_away.min() == MAX_DISTANCE
