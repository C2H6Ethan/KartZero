"""Tests for reading the track layout, using fake memory so that no game file is needed."""

import ctypes

from desmume.mkds.mkds import MAP_DATA_PTR_ADDR, mdat_mapdata_t, struct_nkm_cpoi_entry_t

from kartzero.kart import FX32_ONE
from kartzero.track import Track
from test_kart import FakeMemory

MAP_ADDRESS = 0x02300000
CHECKPOINTS_ADDRESS = 0x02310000


def make_track(values):
    pointers = {
        MAP_DATA_PTR_ADDR: MAP_ADDRESS,
        MAP_ADDRESS + mdat_mapdata_t.cpoi.offset: CHECKPOINTS_ADDRESS,
    }
    return Track(FakeMemory(pointers | values))


def test_checkpoint_count_comes_from_the_map():
    assert make_track({MAP_ADDRESS + mdat_mapdata_t.cpoiCount.offset: 26}).checkpoint_count() == 26


def test_checkpoint_center_is_the_middle_of_its_line():
    entry = CHECKPOINTS_ADDRESS + ctypes.sizeof(struct_nkm_cpoi_entry_t)  # the second checkpoint
    values = {
        entry + struct_nkm_cpoi_entry_t.x1.offset: 0,
        entry + struct_nkm_cpoi_entry_t.z1.offset: 2 * FX32_ONE,
        entry + struct_nkm_cpoi_entry_t.x2.offset: 10 * FX32_ONE,
        entry + struct_nkm_cpoi_entry_t.z2.offset: 4 * FX32_ONE,
    }
    assert make_track(values).checkpoint_center(1) == (5, 3)
