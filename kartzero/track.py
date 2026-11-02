"""Reads the current track's layout out of the game's memory."""

import ctypes
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from desmume.mkds.mkds import (
    COLLISION_DATA_ADDR,
    MAP_DATA_PTR_ADDR,
    kcol_header_t,
    mdat_mapdata_t,
    struct_nkm_cpoi_entry_t,
)

from kartzero.kart import FX32_ONE, Memory

# The game keeps each collision triangle as one corner, a height and some directions.
PRISM_ENTRY = np.dtype(
    [
        ("height", "<i4"),
        ("corner", "<u2"),
        ("face", "<u2"),
        ("edge1", "<u2"),
        ("edge2", "<u2"),
        ("edge3", "<u2"),
        ("attributes", "<u2"),
    ]
)
FIRST_PRISM_OFFSET = 0x10


@dataclass
class CollisionTriangles:
    """Every collision triangle on the track."""

    corners: NDArray[np.float64]  # shape (triangles, 3 corners, x y z)
    is_wall: NDArray[np.bool_]
    is_floor: NDArray[np.bool_]
    floor_type: NDArray[np.integer]


class Track:
    """The track being raced. `memory` is the emulator's memory object, or a fake one in tests."""

    def __init__(self, memory: Memory) -> None:
        self._memory = memory

    def checkpoint_count(self) -> int:
        map_data = self._memory.unsigned.read_long(MAP_DATA_PTR_ADDR)
        return self._memory.unsigned.read_short(map_data + mdat_mapdata_t.cpoiCount.offset)

    def checkpoint_center(self, index: int) -> tuple[float, float]:
        """The middle of a checkpoint line, as (x, z)."""
        map_data = self._memory.unsigned.read_long(MAP_DATA_PTR_ADDR)
        first = self._memory.unsigned.read_long(map_data + mdat_mapdata_t.cpoi.offset)
        entry = first + index * ctypes.sizeof(struct_nkm_cpoi_entry_t)
        x1, z1 = self._read_point(entry + struct_nkm_cpoi_entry_t.x1.offset)
        x2, z2 = self._read_point(entry + struct_nkm_cpoi_entry_t.x2.offset)
        return (x1 + x2) / 2, (z1 + z2) / 2

    def collision_triangles(self) -> CollisionTriangles:
        """Read the collision map. The track never changes, so one read per race start is enough."""
        header = self._read_longs(COLLISION_DATA_ADDR, ctypes.sizeof(kcol_header_t) // 4)
        positions = int(header[kcol_header_t.posDataOffset.offset // 4])
        normals = int(header[kcol_header_t.nrmDataOffset.offset // 4])
        prisms = int(header[kcol_header_t.prismDataOffset.offset // 4])
        blocks = int(header[kcol_header_t.blockDataOffset.offset // 4])

        count = (blocks - prisms - FIRST_PRISM_OFFSET) // PRISM_ENTRY.itemsize
        entries = np.frombuffer(
            self._read_longs(prisms + FIRST_PRISM_OFFSET, count * PRISM_ENTRY.itemsize // 4).tobytes(), PRISM_ENTRY
        )
        vertex_count = int(entries["corner"].max()) + 1
        vertices = np.frombuffer(self._read_longs(positions, vertex_count * 3).tobytes(), "<i4").reshape(-1, 3)
        normal_count = max(int(entries[name].max()) for name in ("face", "edge1", "edge2", "edge3")) + 1
        directions = np.frombuffer(self._read_longs(normals, (normal_count * 3 + 1) // 2).tobytes(), "<i2")
        directions = directions[: normal_count * 3].reshape(-1, 3) / FX32_ONE

        face, edge1, edge2, edge3 = (directions[entries[name]] for name in ("face", "edge1", "edge2", "edge3"))
        along1, along2 = np.cross(edge1, face), np.cross(edge2, face)
        height = (entries["height"] / FX32_ONE)[:, None]
        first = vertices[entries["corner"]] / FX32_ONE
        second = first + along2 * height / np.sum(edge3 * along2, axis=1, keepdims=True)
        third = first + along1 * height / np.sum(edge3 * along1, axis=1, keepdims=True)

        attributes = entries["attributes"]
        return CollisionTriangles(
            corners=np.stack([first, second, third], axis=1),
            is_wall=((attributes >> 14) & 1).astype(bool),
            is_floor=((attributes >> 15) & 1).astype(bool),
            floor_type=(attributes >> 8) & 0x1F,
        )

    def _read_longs(self, address: int, count: int) -> NDArray[np.uint32]:
        read = self._memory.unsigned.read_long
        return np.array([read(address + index * 4) for index in range(count)], dtype=np.uint32)

    def _read_point(self, address: int) -> tuple[float, float]:
        # x and z sit next to each other in memory
        return (
            self._memory.signed.read_long(address) / FX32_ONE,
            self._memory.signed.read_long(address + 4) / FX32_ONE,
        )
