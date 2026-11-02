"""Reads live kart and race data out of the game's memory."""

from desmume.mkds.mkds import (
    RACE_CONFIG_PTR_ADDR,
    RACE_STATUS_PTR_ADDR,
    RACER_PTR_ADDR,
    driver_t,
    race_driver_status_t,
    race_status_t,
)

FX32_ONE = 4096  # the game stores decimals as whole numbers: 4096 means 1.0
FINISH_PROGRESS = 1.0  # race_progress at the finish line
PLAYER_CONFIG_OFFSET = 0x68
PLAYER_ENTRY_OFFSET = race_status_t.driverStatus.offset  # our kart is the first entry


class Kart:
    """The player's kart. Every method reads the current value from memory.

    `memory` is the emulator's memory object, or a fake one in tests.
    """

    def __init__(self, memory):
        self._memory = memory

    def speed(self):
        """Distance covered per frame."""
        return self._read_racer_decimal(driver_t.speed.offset)

    def position(self):
        """Where the kart is on the track, as (x, y, z)."""
        start = driver_t.position.offset
        return tuple(self._read_racer_decimal(start + axis * 4) for axis in range(3))

    def direction(self):
        """Which way the kart points, as (x, y, z)."""
        start = driver_t.direction.offset
        return tuple(self._read_racer_decimal(start + axis * 4) for axis in range(3))

    def race_progress(self):
        """How far through the whole race: each of the 3 laps adds a third, and behind the start line it's negative."""
        return self._read_status_decimal(race_driver_status_t.raceProgress.offset)

    def has_finished(self):
        """Whether the race is over. The game then drives the kart itself, so a run must end here."""
        return self.race_progress() >= FINISH_PROGRESS

    def lap_progress(self):
        """How far around the current lap. Jumps between 0 and 1 at the line, so don't use it for rewards."""
        return self._read_status_decimal(race_driver_status_t.lapProgress.offset)

    def checkpoint_progress(self):
        """How far along the current section between checkpoints."""
        return self._read_status_decimal(race_driver_status_t.cpoiProgress.offset)

    def checkpoint(self):
        """Which checkpoint the kart is at."""
        return self._memory.unsigned.read_short(self._status_field(race_driver_status_t.curCpoi.offset))

    def lap(self):
        """Which lap the kart is on."""
        return self._memory.signed.read_long(self._status_field(race_driver_status_t.curLap.offset))

    def driver_id(self):
        """Which character is driving."""
        return self._memory.unsigned.read_byte(self._player_config())

    def kart_id(self):
        """Which kart is being driven."""
        return self._memory.unsigned.read_byte(self._player_config() + 4)

    def is_loaded(self):
        """Whether a kart exists in memory. It can briefly be wrong while a race loads."""
        return self._memory.unsigned.read_long(RACER_PTR_ADDR) != 0

    def _read_racer_decimal(self, offset):
        racer = self._memory.unsigned.read_long(RACER_PTR_ADDR)
        return self._memory.signed.read_long(racer + offset) / FX32_ONE

    def _read_status_decimal(self, offset):
        return self._memory.signed.read_long(self._status_field(offset)) / FX32_ONE

    def _status_field(self, offset):
        status = self._memory.unsigned.read_long(RACE_STATUS_PTR_ADDR)
        return status + PLAYER_ENTRY_OFFSET + offset

    def _player_config(self):
        return self._memory.unsigned.read_long(RACE_CONFIG_PTR_ADDR) + PLAYER_CONFIG_OFFSET
