"""Tests for reading kart data, using fake memory so that no game file is needed."""

import pytest
from desmume.mkds.mkds import (
    RACE_CONFIG_PTR_ADDR,
    RACE_STATUS_PTR_ADDR,
    RACER_PTR_ADDR,
    driver_t,
    race_driver_status_t,
    race_status_t,
)

from kartzero.kart import FX32_ONE, Kart

RACER_ADDRESS = 0x02300000
STATUS_ADDRESS = 0x02310000
CONFIG_ADDRESS = 0x02320000
PLAYER_ADDRESS = STATUS_ADDRESS + race_status_t.driverStatus.offset


class FakeReader:
    def __init__(self, values):
        self._values = values

    def read_byte(self, address):
        return self._values[address]

    def read_short(self, address):
        return self._values[address]

    def read_long(self, address):
        return self._values[address]


class FakeMemory:
    def __init__(self, values):
        self.unsigned = FakeReader(values)
        self.signed = FakeReader(values)


def make_kart(values=None):
    pointers = {
        RACER_PTR_ADDR: RACER_ADDRESS,
        RACE_STATUS_PTR_ADDR: STATUS_ADDRESS,
        RACE_CONFIG_PTR_ADDR: CONFIG_ADDRESS,
    }
    return Kart(FakeMemory(pointers | (values or {})))


def test_speed_follows_the_racer_pointer_and_converts_to_a_decimal():
    address = RACER_ADDRESS + driver_t.speed.offset
    assert make_kart({address: int(7.5 * FX32_ONE)}).speed() == 7.5


def test_position_reads_x_y_z_in_order():
    start = RACER_ADDRESS + driver_t.position.offset
    values = {start: 1 * FX32_ONE, start + 4: 2 * FX32_ONE, start + 8: -3 * FX32_ONE}
    assert make_kart(values).position() == (1, 2, -3)


def test_direction_reads_x_y_z_in_order():
    start = RACER_ADDRESS + driver_t.direction.offset
    values = {start: FX32_ONE, start + 4: 0, start + 8: -FX32_ONE}
    assert make_kart(values).direction() == (1, 0, -1)


@pytest.mark.parametrize(
    "method, field",
    [
        ("race_progress", race_driver_status_t.raceProgress),
        ("lap_progress", race_driver_status_t.lapProgress),
        ("checkpoint_progress", race_driver_status_t.cpoiProgress),
    ],
)
def test_progress_reads_from_our_entry_and_can_be_negative(method, field):
    kart = make_kart({PLAYER_ADDRESS + field.offset: -FX32_ONE // 2})
    assert getattr(kart, method)() == -0.5


@pytest.mark.parametrize("progress, finished", [(0.9999, False), (1.0, True), (1.0002, True)])
def test_has_finished_at_the_line_and_past_it(progress, finished):
    address = PLAYER_ADDRESS + race_driver_status_t.raceProgress.offset
    assert make_kart({address: int(progress * FX32_ONE)}).has_finished() == finished


def test_checkpoint_and_lap_are_whole_numbers():
    values = {
        PLAYER_ADDRESS + race_driver_status_t.curCpoi.offset: 12,
        PLAYER_ADDRESS + race_driver_status_t.curLap.offset: 2,
    }
    kart = make_kart(values)
    assert kart.checkpoint() == 12
    assert kart.lap() == 2


def test_driver_and_kart_come_from_the_player_config():
    values = {CONFIG_ADDRESS + 0x68: 8, CONFIG_ADDRESS + 0x6C: 34}
    kart = make_kart(values)
    assert kart.driver_id() == 8
    assert kart.kart_id() == 34


def test_is_loaded_needs_a_racer_pointer():
    assert make_kart().is_loaded()
    assert not make_kart({RACER_PTR_ADDR: 0}).is_loaded()
