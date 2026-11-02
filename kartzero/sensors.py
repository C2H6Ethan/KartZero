"""Measures how far the kart can see in each direction, like a set of laser rangefinders."""

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image, ImageDraw

from kartzero.kart import OFF_ROAD_FLOOR_TYPES, Vector3
from kartzero.track import CollisionTriangles

MAX_DISTANCE = 1000.0  # anything further away reads as this
RAY_ANGLES = tuple(math.radians(angle) for angle in (-90, -60, -30, -10, 0, 10, 30, 60, 90))
RAY_HEIGHT = 10.0  # rays start a little above the kart's position
GRID_CELL_SIZE = 4.0  # the kart moves about 7.7 per frame

Numbers = NDArray[np.float64]


class Lines:
    """Flat line pieces on the track, each with the height range where it blocks a ray."""

    def __init__(self, starts: ArrayLike, ends: ArrayLike, lowest: ArrayLike, highest: ArrayLike) -> None:
        self.starts = np.asarray(starts, dtype=float).reshape(-1, 2)
        self.ends = np.asarray(ends, dtype=float).reshape(-1, 2)
        self.lowest = np.asarray(lowest, dtype=float)
        self.highest = np.asarray(highest, dtype=float)

    def distances(self, x: float, z: float, y: float, angles: Numbers) -> Numbers:
        """How far each ray from (x, z) travels before hitting a line at this height."""
        result = np.full(len(angles), MAX_DISTANCE)
        active = (self.lowest <= y) & (y <= self.highest)
        if not active.any():
            return result
        rays = np.column_stack([np.cos(angles), np.sin(angles)])
        starts = self.starts[active] - np.array([x, z])
        spans = self.ends[active] - self.starts[active]

        # Solve "ray point = line point" for every pair of ray and line at once.
        cross = rays[:, 0, None] * spans[:, 1] - rays[:, 1, None] * spans[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            along_ray = (starts[:, 0] * spans[:, 1] - starts[:, 1] * spans[:, 0]) / cross
            along_line = (starts[:, 0] * rays[:, 1, None] - starts[:, 1] * rays[:, 0, None]) / cross
        hits = (along_ray > 0) & (along_line >= 0) & (along_line <= 1)
        along_ray = np.where(hits, along_ray, MAX_DISTANCE)
        return np.minimum(along_ray.min(axis=1), MAX_DISTANCE)


def wall_lines(triangles: CollisionTriangles) -> Lines:
    """Walls stand upright, so seen from above each triangle is a line between its furthest corners."""
    corners = triangles.corners[triangles.is_wall]
    flat = corners[:, :, [0, 2]]
    pairs = [(0, 1), (1, 2), (0, 2)]
    lengths = np.stack([np.linalg.norm(flat[:, a] - flat[:, b], axis=1) for a, b in pairs], axis=1)
    longest = lengths.argmax(axis=1)
    rows = np.arange(len(flat))
    first = np.array([pairs[index][0] for index in longest])
    second = np.array([pairs[index][1] for index in longest])
    return Lines(flat[rows, first], flat[rows, second], corners[:, :, 1].min(axis=1), corners[:, :, 1].max(axis=1))


class RoadGrid:
    """A top-down picture of the track, where each square is road or not.

    Painted once, so checking a square is instant. It ignores height, so where the
    track crosses itself the road on either level counts as road.
    """

    def __init__(self, triangles: CollisionTriangles, cell_size: float = GRID_CELL_SIZE) -> None:
        self.cell_size = cell_size
        is_road = triangles.is_floor & ~np.isin(triangles.floor_type, list(OFF_ROAD_FLOOR_TYPES))
        flat = triangles.corners[is_road][:, :, [0, 2]]
        self.origin = flat.reshape(-1, 2).min(axis=0) - cell_size
        cells = ((flat.reshape(-1, 2).max(axis=0) - self.origin) / cell_size).astype(int) + 2
        picture = Image.new("1", (int(cells[0]), int(cells[1])))
        draw = ImageDraw.Draw(picture)
        for triangle in (flat - self.origin) / cell_size:
            draw.polygon([(float(x), float(z)) for x, z in triangle], fill=1)
        self.cells = np.array(picture, dtype=bool)  # indexed as [z, x]

    def is_road(self, x: ArrayLike, z: ArrayLike) -> NDArray[np.bool_]:
        """Whether each point is on the road. Anywhere off the picture isn't."""
        column = np.floor((np.asarray(x) - self.origin[0]) / self.cell_size).astype(int)
        row = np.floor((np.asarray(z) - self.origin[1]) / self.cell_size).astype(int)
        inside = (row >= 0) & (row < self.cells.shape[0]) & (column >= 0) & (column < self.cells.shape[1])
        return inside & self.cells[row.clip(0, self.cells.shape[0] - 1), column.clip(0, self.cells.shape[1] - 1)]

    def distances(self, x: float, z: float, angles: Numbers) -> Numbers:
        """How far each ray travels before the surface changes: road to not road, or the other way."""
        steps = np.arange(self.cell_size, MAX_DISTANCE, self.cell_size)
        along_x = x + np.cos(angles)[:, None] * steps
        along_z = z + np.sin(angles)[:, None] * steps
        changed = self.is_road(along_x, along_z) != self.is_road(x, z)
        first = changed.argmax(axis=1)
        return np.where(changed.any(axis=1), steps[first], MAX_DISTANCE)


class TrackSensors:
    """How far the walls and the edges of the road are, from the kart's point of view."""

    def __init__(self, triangles: CollisionTriangles) -> None:
        self.walls = wall_lines(triangles)
        self.road = RoadGrid(triangles)

    def wall_distances(self, position: Vector3, heading: float) -> Numbers:
        """Distance to the nearest wall along each ray in RAY_ANGLES."""
        x, y, z = position
        return self.walls.distances(x, z, y + RAY_HEIGHT, heading + np.array(RAY_ANGLES))

    def road_edge_distances(self, position: Vector3, heading: float) -> Numbers:
        """Distance to where the surface changes along each ray in RAY_ANGLES.

        On the road that's where the road ends. Off the road it's where the road starts.
        """
        x, _, z = position
        return self.road.distances(x, z, heading + np.array(RAY_ANGLES))
