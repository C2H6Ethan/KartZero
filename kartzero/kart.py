"""Reads live kart and race data out of the game's memory."""

import math
from typing import Callable, Protocol

from desmume.mkds.mkds import (
    RACE_CONFIG_PTR_ADDR,
    RACE_STATUS_PTR_ADDR,
    RACER_PTR_ADDR,
    driver_t,
    race_driver_status_t,
    race_status_t,
    race_status_time_t,
    race_time_t,
)

FX32_ONE = 4096  # the game stores decimals as whole numbers: 4096 means 1.0
FINISH_PROGRESS = 1.0  # race_progress at the finish line
DEGREES_PER_ANGLE_UNIT = 360 / 65536  # the game divides a full turn into 65536
OFF_ROAD_FLOOR_TYPES = {3}  # measured on Figure-8: other tracks may add more
IN_THE_AIR = -1
PLAYER_CONFIG_OFFSET = 0x68
PLAYER_ENTRY_OFFSET = race_status_t.driverStatus.offset  # our kart is the first entry
RACE_CLOCK_OFFSET = race_status_t.time.offset + race_status_time_t.lapTime.offset

Vector3 = tuple[float, float, float]


class MemoryReader(Protocol):
    """Reads numbers of different sizes at an address."""

    def read_byte(self, address: int) -> int: ...

    def read_short(self, address: int) -> int: ...

    def read_long(self, address: int) -> int: ...


class Memory(Protocol):
    """The game's memory: the emulator's own memory object, or a fake one in tests."""

    @property
    def unsigned(self) -> MemoryReader: ...

    @property
    def signed(self) -> MemoryReader: ...


class Kart:
    """The player's kart. Every method reads the current value from memory."""

    def __init__(self, memory: Memory) -> None:
        self._memory = memory

    def speed(self) -> float:
        """Distance covered per frame."""
        return self._read_racer_decimal(driver_t.speed.offset)

    def speed_fraction(self) -> float:
        """How close the kart is to its current top speed: 1.0 at top speed, 0 standing still.

        A boost raises the top speed first, so this dips when a boost starts, then climbs back to 1.0.
        """
        return self._read_racer_decimal(driver_t.maxSpeedFraction.offset)

    def position(self) -> Vector3:
        """Where the kart is on the track, as (x, y, z)."""
        return self._read_racer_vector(driver_t.position.offset)

    def direction(self) -> Vector3:
        """Which way the kart points, as (x, y, z)."""
        return self._read_racer_vector(driver_t.direction.offset)

    def heading(self) -> float:
        """Which way the kart points on the track, as an angle in radians."""
        x, _, z = self.direction()
        return math.atan2(z, x)

    def turning_rate(self) -> int:
        """How fast the kart is turning. Negative and positive are opposite directions."""
        return self._read_racer(self._memory.signed.read_long, driver_t.yRotSpeed.offset)

    def drift_angle(self) -> float:
        """How far the kart is angled in a drift, in degrees: 0 is no drift, the sign is the side."""
        return self._read_racer(self._memory.signed.read_short, driver_t.driftRotY.offset) * DEGREES_PER_ANGLE_UNIT

    def mini_turbo_count(self) -> int:
        """How many left and right steers count towards a mini-turbo. Resets when the drift ends."""
        return self._read_racer(self._memory.signed.read_long, driver_t.driftLeftRightCount.offset)

    def boost_time_left(self) -> int:
        """How many frames of boost are left."""
        return self._read_racer(self._memory.signed.read_short, driver_t.boostTimer.offset)

    def is_off_road(self) -> bool:
        """Whether the kart is on a slowing surface like grass."""
        return self._read_racer(self._memory.unsigned.read_long, driver_t.floorColType.offset) in OFF_ROAD_FLOOR_TYPES

    def is_on_ground(self) -> bool:
        """Whether the kart is touching the ground, rather than in the air."""
        return self._read_racer(self._memory.signed.read_long, driver_t.floorDriverColType.offset) != IN_THE_AIR

    def race_progress(self) -> float:
        """How far through the whole race: each of the 3 laps adds a third, and behind the start line it's negative."""
        return self._read_status_decimal(race_driver_status_t.raceProgress.offset)

    def has_finished(self) -> bool:
        """Whether the race is over. The game then drives the kart itself, so a run must end here."""
        return self.race_progress() >= FINISH_PROGRESS

    def race_clock(self) -> float:
        """The race time shown on screen, in seconds. It starts in the countdown, about 2.4 s
        before the kart can move, and keeps running after the finish."""
        clock = self._memory.unsigned.read_long(RACE_STATUS_PTR_ADDR) + RACE_CLOCK_OFFSET
        minutes = self._memory.unsigned.read_byte(clock + race_time_t.minutes.offset)
        seconds = self._memory.unsigned.read_byte(clock + race_time_t.seconds.offset)
        milliseconds = self._memory.unsigned.read_short(clock + race_time_t.milliseconds.offset)
        return minutes * 60 + seconds + milliseconds / 1000

    def finish_time(self) -> float:
        """The official race time in seconds, as the game records it at the line. 0 until then."""
        return self._memory.unsigned.read_long(self._status_field(race_driver_status_t.totalMilliseconds.offset)) / 1000

    def lap_progress(self) -> float:
        """How far around the current lap. Jumps between 0 and 1 at the line, so don't use it for rewards."""
        return self._read_status_decimal(race_driver_status_t.lapProgress.offset)

    def checkpoint_progress(self) -> float:
        """How far along the current section between checkpoints."""
        return self._read_status_decimal(race_driver_status_t.cpoiProgress.offset)

    def checkpoint(self) -> int:
        """Which checkpoint the kart is at."""
        return self._memory.unsigned.read_short(self._status_field(race_driver_status_t.curCpoi.offset))

    def lap(self) -> int:
        """Which lap the kart is on."""
        return self._memory.signed.read_long(self._status_field(race_driver_status_t.curLap.offset))

    def driver_id(self) -> int:
        """Which character is driving."""
        return self._memory.unsigned.read_byte(self._player_config())

    def kart_id(self) -> int:
        """Which kart is being driven."""
        return self._memory.unsigned.read_byte(self._player_config() + 4)

    def is_loaded(self) -> bool:
        """Whether a kart exists in memory. It can briefly be wrong while a race loads."""
        return self._memory.unsigned.read_long(RACER_PTR_ADDR) != 0

    def _read_racer(self, read: Callable[[int], int], offset: int) -> int:
        return read(self._memory.unsigned.read_long(RACER_PTR_ADDR) + offset)

    def _read_racer_decimal(self, offset: int) -> float:
        return self._read_racer(self._memory.signed.read_long, offset) / FX32_ONE

    def _read_racer_vector(self, offset: int) -> Vector3:
        return (
            self._read_racer_decimal(offset),
            self._read_racer_decimal(offset + 4),
            self._read_racer_decimal(offset + 8),
        )

    def _read_status_decimal(self, offset: int) -> float:
        return self._memory.signed.read_long(self._status_field(offset)) / FX32_ONE

    def _status_field(self, offset: int) -> int:
        status = self._memory.unsigned.read_long(RACE_STATUS_PTR_ADDR)
        return status + PLAYER_ENTRY_OFFSET + offset

    def _player_config(self) -> int:
        return self._memory.unsigned.read_long(RACE_CONFIG_PTR_ADDR) + PLAYER_CONFIG_OFFSET
