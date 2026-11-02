"""Check the kart's state readings and distance sensors on the real game.

Forces a mini-turbo on the start straight, crashes into a wall, compares the road grid with
the game's own floor type during a race, measures the sensors' speed, and draws a map.
Run make_start_state.py first.

Usage: python scripts/sensor_check.py OUT_IMAGE.png
"""

import math
import sys
import time
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from kartzero.game import Button, Game
from kartzero.kart import Kart, Vector3
from kartzero.sensors import MAX_DISTANCE, RAY_ANGLES, Numbers, TrackSensors
from kartzero.test_driver import TestDriver

FRAMES_PER_DECISION = 4
TIMING_DECISIONS = 3000
CRASH_SPEED_DROP = 2
IMAGE_SCALE = 2

SensorSample = tuple[Vector3, float, Numbers, Numbers]


def show_kart(kart: Kart, tag: str) -> None:
    print(
        f"  {tag:8s} speed={kart.speed():6.2f} drift_angle={kart.drift_angle():6.1f} turning={kart.turning_rate():5d} "
        f"count={kart.mini_turbo_count()} boost={kart.boost_time_left():3d} off_road={kart.is_off_road()} ground={kart.is_on_ground()}",
        flush=True,
    )


def force_mini_turbo(game: Game) -> None:
    print("\nMini-turbo: drift left, alternate, release")
    game.start_race()
    game.advance(120)
    show_kart(game.kart, "straight")
    game.hold(Button.A | Button.R)
    game.advance(2)
    show_kart(game.kart, "hop")
    game.hold(Button.A | Button.R | Button.LEFT)
    game.advance(40)
    show_kart(game.kart, "drift")
    for index in range(4):
        game.hold(Button.A | Button.R | (Button.RIGHT if index % 2 == 0 else Button.LEFT))
        game.advance(16)
        show_kart(game.kart, f"steer{index}")
    game.hold(Button.A)
    for index in range(4):
        game.advance()
        show_kart(game.kart, f"release{index}")


def crash_into_walls(game: Game, sensors: TrackSensors) -> None:
    print("\nCrashes: the nearest wall at the moment the kart suddenly slows")
    for turn in (Button.LEFT, Button.RIGHT):
        game.start_race()
        game.advance(60)
        game.hold(Button.A | turn)
        speed = game.kart.speed()
        for _ in range(120):
            game.advance(FRAMES_PER_DECISION)
            before, speed = speed, game.kart.speed()
            if before - speed > CRASH_SPEED_DROP:
                walls = sensors.wall_distances(game.kart.position(), game.kart.heading())
                print(f"  {turn.name}: speed {before:.2f} -> {speed:.2f}, nearest wall {walls.min():.1f}")


def race_with_sensors(game: Game, sensors: TrackSensors) -> tuple[list[Vector3], SensorSample]:
    print("\nFull race with the test driver")
    game.start_race()
    driver = TestDriver(game)
    path: list[Vector3] = []
    on_road: list[float] = []
    off_road: list[float] = []
    sample: Optional[SensorSample] = None
    agree = false_walls = 0
    while not game.kart.has_finished():
        driver.drive()
        game.advance(FRAMES_PER_DECISION)
        position, heading = game.kart.position(), game.kart.heading()
        path.append(position)
        walls = sensors.wall_distances(position, heading)
        edges = sensors.road_edge_distances(position, heading)
        # The kart never hit anything in this race, so a wall touching it would be a sensor mistake.
        false_walls += int(walls.min() < 5)
        is_off_road = game.kart.is_off_road()
        agree += bool(sensors.road.is_road(np.array(position[0]), np.array(position[2]))) != is_off_road
        (off_road if is_off_road else on_road).append(edges.min())
        if len(path) == 200:
            sample = (position, heading, walls, edges)
    if sample is None:
        raise RuntimeError("the race ended before the sample at decision 200")
    print(f"  decisions {len(path)}, finished {game.kart.has_finished()}")
    print(f"  road grid agrees with the game's floor type: {agree} of {len(path)} ({agree / len(path):.1%})")
    print(f"  walls touching the kart when it wasn't crashing: {false_walls} times")
    print(f"  nearest surface change, on road: median {np.median(on_road):.0f}, off road: median {np.median(off_road):.0f}")
    print(f"  sample at decision 200: walls {np.round(sample[2]).tolist()}, surface changes {np.round(sample[3]).tolist()}")
    return path, sample


def measure_cost(game: Game, sensors: TrackSensors) -> None:
    frames = TIMING_DECISIONS * FRAMES_PER_DECISION
    game.start_race()
    start = time.perf_counter()
    for _ in range(TIMING_DECISIONS):
        game.advance(FRAMES_PER_DECISION)
    without = frames / (time.perf_counter() - start)
    game.start_race()
    start = time.perf_counter()
    for _ in range(TIMING_DECISIONS):
        game.advance(FRAMES_PER_DECISION)
        position, heading = game.kart.position(), game.kart.heading()
        sensors.wall_distances(position, heading)
        sensors.road_edge_distances(position, heading)
    print(f"\nSpeed: {without:.0f} frames/s alone, {frames / (time.perf_counter() - start):.0f} with both sensors every decision")


def draw_map(out: str, sensors: TrackSensors, path: list[Vector3], sample: SensorSample) -> None:
    grid = sensors.road
    image = Image.fromarray(np.where(grid.cells, 190, 255).astype(np.uint8)).convert("RGB")
    image = image.resize((image.width // IMAGE_SCALE, image.height // IMAGE_SCALE))
    draw = ImageDraw.Draw(image)
    
    def pixel(x: float, z: float) -> tuple[float, float]:
        return (float(x - grid.origin[0]) / grid.cell_size / IMAGE_SCALE, float(z - grid.origin[1]) / grid.cell_size / IMAGE_SCALE)

    for start, end in zip(sensors.walls.starts, sensors.walls.ends):
        draw.line([pixel(*start), pixel(*end)], fill=(210, 30, 30))
    draw.line([pixel(x, z) for x, _, z in path], fill=(30, 170, 60))
    (x, _, z), heading, walls, edges = sample
    for angle, wall, edge in zip(RAY_ANGLES, walls, edges):
        direction = np.array([math.cos(heading + angle), math.sin(heading + angle)])
        draw.line([pixel(x, z), pixel(*(np.array([x, z]) + direction * wall))], fill=(0, 0, 0))
        draw.line([pixel(x, z), pixel(*(np.array([x, z]) + direction * edge))], fill=(40, 110, 240), width=2)
    image.save(out)
    print(f"\nMap saved: grey road, red walls, green path, black wall rays, blue surface rays. Rays stop at {MAX_DISTANCE:.0f}.")


def main() -> None:
    out = sys.argv[1]
    with Game() as game:
        force_mini_turbo(game)
        game.load_snapshot()
        sensors = TrackSensors(game.track.collision_triangles())
        crash_into_walls(game, sensors)
        path, sample = race_with_sensors(game, sensors)
        measure_cost(game, sensors)
        draw_map(out, sensors, path, sample)


if __name__ == "__main__":
    main()
