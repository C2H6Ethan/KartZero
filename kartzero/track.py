"""Reads the current track's layout out of the game's memory."""

import ctypes

from desmume.mkds.mkds import MAP_DATA_PTR_ADDR, mdat_mapdata_t, struct_nkm_cpoi_entry_t

from kartzero.kart import FX32_ONE


class Track:
    """The track being raced. `memory` is the emulator's memory object, or a fake one in tests."""

    def __init__(self, memory):
        self._memory = memory

    def checkpoint_count(self):
        map_data = self._memory.unsigned.read_long(MAP_DATA_PTR_ADDR)
        return self._memory.unsigned.read_short(map_data + mdat_mapdata_t.cpoiCount.offset)

    def checkpoint_center(self, index):
        """The middle of a checkpoint line, as (x, z)."""
        map_data = self._memory.unsigned.read_long(MAP_DATA_PTR_ADDR)
        first = self._memory.unsigned.read_long(map_data + mdat_mapdata_t.cpoi.offset)
        entry = first + index * ctypes.sizeof(struct_nkm_cpoi_entry_t)
        x1, z1 = self._read_point(entry + struct_nkm_cpoi_entry_t.x1.offset)
        x2, z2 = self._read_point(entry + struct_nkm_cpoi_entry_t.x2.offset)
        return (x1 + x2) / 2, (z1 + z2) / 2

    def _read_point(self, address):
        # x and z sit next to each other in memory
        return (
            self._memory.signed.read_long(address) / FX32_ONE,
            self._memory.signed.read_long(address + 4) / FX32_ONE,
        )
