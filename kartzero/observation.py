"""Turns the game's current state into the list of numbers the AI receives each decision."""

import math

import numpy as np

from kartzero.game import Button
from kartzero.track import Track
from kartzero.sensors import MAX_DISTANCE, TrackSensors
from kartzero.kart import Kart, Vector3

# Dividing by these keeps each number around -1 to 1, which networks learn from best.
SPEED_SCALE = 10.0  # measured top speed with a boost is 10.04
DRIFT_ANGLE_SCALE = 60.0  # measured full drift angle is about 60 degrees
TURNING_SCALE = 300.0  # measured turning rate reached about 300
MINI_TURBO_SCALE = 4.0  # a mini-turbo was ready at a count of 4
BOOST_SCALE = 55.0  # a mini-turbo gave 55 frames of boost
COUNTDOWN_SCALE = 3.0  # races start about 3 seconds before GO

BUTTONS = (Button.A, Button.R, Button.LEFT, Button.RIGHT)
CHECKPOINTS_AHEAD = (1, 2, 3)


def view_of_point(position: Vector3, heading: float, point: tuple[float, float]) -> tuple[float, float]:
    """Describe a point on the track from the kart's point of view.

    `position` is the kart's (x, y, z), `heading` is the angle it points in radians, and `point` is (x, z).
    Returns (distance, angle). The angle is in radians from -pi to pi, and 0 is straight ahead.
    """
    dx = point[0] - position[0]
    dz = point[1] - position[2]

    distance = math.sqrt(dx**2 + dz**2)

    map_direction = math.atan2(dz, dx)
    angle = map_direction - heading

    if angle > math.pi:
        angle -=  2 * math.pi
    if angle < -math.pi:
        angle += 2 * math.pi

    return (distance, angle)


def observe(
    kart: Kart, track: Track, sensors: TrackSensors, held_buttons: Button, seconds_until_go: float
) -> np.ndarray:
    """Build the numbers the AI receives, in exactly this order:

    1. speed / SPEED_SCALE
    2. drift angle / DRIFT_ANGLE_SCALE
    3. turning rate / TURNING_SCALE
    4. mini-turbo count / MINI_TURBO_SCALE
    5. boost time left / BOOST_SCALE
    6. 1 if off road, otherwise 0
    7. 1 if on the ground, otherwise 0
    8. for each of BUTTONS: 1 if it's in held_buttons, otherwise 0
    9. each wall distance / MAX_DISTANCE
    10. each road edge distance / MAX_DISTANCE
    11. for each of CHECKPOINTS_AHEAD: the distance to that checkpoint / MAX_DISTANCE, then its angle / pi
    12. seconds until GO / COUNTDOWN_SCALE, as the countdown on screen shows, and 0 once the race is on

    Returns a numpy array of type float32.
    """
    position, heading = kart.position(), kart.heading()

    numbers: list[float] = [
        kart.speed() / SPEED_SCALE,
        kart.drift_angle() / DRIFT_ANGLE_SCALE,
        kart.turning_rate() / TURNING_SCALE,
        kart.mini_turbo_count() / MINI_TURBO_SCALE,
        kart.boost_time_left() / BOOST_SCALE,
        1 if kart.is_off_road() else 0,
        1 if kart.is_on_ground() else 0,
    ]

    for b in BUTTONS:
        if b & held_buttons:
            numbers.append(1)
        else:
            numbers.append(0)

    wall_distances = sensors.wall_distances(position, heading)
    numbers.extend(wall_distances / MAX_DISTANCE)

    road_edge_distances = sensors.road_edge_distances(position, heading)
    numbers.extend(road_edge_distances / MAX_DISTANCE)

    for ahead in CHECKPOINTS_AHEAD:
        checkpoint = (kart.checkpoint() + ahead) % track.checkpoint_count()

        checkpoint_pos = track.checkpoint_center(checkpoint)

        distance, angle = view_of_point(position, heading, checkpoint_pos)

        numbers.append(distance / MAX_DISTANCE)
        numbers.append(angle / math.pi)

    numbers.append(seconds_until_go / COUNTDOWN_SCALE)

    return np.array(numbers, dtype=np.float32)
