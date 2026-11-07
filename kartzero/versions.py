"""Keeps past versions of the AI to watch later, without ending up with thousands of them.

Versions are close together early on, when the AI changes fastest, and further apart later:
each one waits 3% longer than the last, and at least about a minute. That gives about 200
versions by 100 million decisions.
"""

import csv
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

SMALLEST_GAP = 25_000  # about a minute of training
GROWTH_PER_VERSION = 1.03
INDEX_FILE = "versions.csv"
TEST_LAPS_FILE = "test_laps.csv"
# Mini-turbos counted afterwards by scripts/lap_boosts.py, for laps tested before the lap tester
# counted them. The test lap is the same every time, so the count matches the lap exactly.
MINI_TURBO_COUNTS_FILE = "mini_turbo_counts.csv"
FIELDS = ["file", "decisions", "minutes", "average_end_progress"]
TEST_LAP_FIELDS = ["file", "finished", "seconds", "progress", "mini_turbos"]


def training_time(minutes: float) -> str:
    """Total time spent training, like "0030h 12m". Time while training is stopped doesn't count."""
    whole_minutes = int(minutes)
    return f"{whole_minutes // 60:04d}h {whole_minutes % 60:02d}m"


def next_version_after(decisions: int) -> int:
    """How many decisions the next version should be saved at."""
    due = float(SMALLEST_GAP)
    while due <= decisions:
        due *= GROWTH_PER_VERSION
    return max(math.ceil(due), decisions + SMALLEST_GAP)


@dataclass
class Version:
    """One saved copy of the AI."""

    file: Path
    decisions: int
    minutes: float
    average_end_progress: float


@dataclass
class LapResult:
    """An official test lap of one version."""

    finished: bool
    seconds: float
    progress: float
    mini_turbos: Optional[int] = None  # None for laps recorded before they were counted

    def description(self) -> str:
        """A race time if it finished, otherwise how far it got."""
        if not self.finished:
            return f"{self.progress:.1%}"
        minutes, seconds = divmod(self.seconds, 60)
        return f"{int(minutes)}:{seconds:05.2f}"


class VersionStore:
    """The saved versions of one training run, in its folder, with an index you can open in Excel."""

    def __init__(self, run_folder: Path) -> None:
        self._folder = run_folder / "versions"
        self._index = self._folder / INDEX_FILE
        self._test_laps = self._folder / TEST_LAPS_FILE
        self._mini_turbo_counts = self._folder / MINI_TURBO_COUNTS_FILE

    def save_test_lap(self, version_file: Path, result: LapResult) -> None:
        self._upgrade_test_laps_file()
        is_new_file = not self._test_laps.exists()
        with open(self._test_laps, "a", newline="") as file:
            writer = csv.writer(file)
            if is_new_file:
                writer.writerow(TEST_LAP_FIELDS)
            mini_turbos = "" if result.mini_turbos is None else result.mini_turbos
            writer.writerow(
                [version_file.name, result.finished, f"{result.seconds:.3f}", f"{result.progress:.4f}", mini_turbos]
            )

    def _upgrade_test_laps_file(self) -> None:
        """Files from before mini-turbos were counted get the new column, empty for their old laps."""
        if not self._test_laps.exists():
            return
        with open(self._test_laps, newline="") as file:
            rows = list(csv.reader(file))
        if not rows or rows[0] == TEST_LAP_FIELDS:
            return
        missing = len(TEST_LAP_FIELDS) - len(rows[0])
        temporary = self._test_laps.with_suffix(".tmp")
        with open(temporary, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(TEST_LAP_FIELDS)
            writer.writerows(row + [""] * missing for row in rows[1:])
        os.replace(temporary, self._test_laps)

    def test_laps(self) -> dict[str, LapResult]:
        """The test lap of each version that has one, by file name, with its mini-turbos if counted."""
        if not self._test_laps.exists():
            return {}
        with open(self._test_laps, newline="") as file:
            laps = {
                row["file"]: LapResult(
                    row["finished"] == "True",
                    float(row["seconds"]),
                    float(row["progress"]),
                    int(row["mini_turbos"]) if row.get("mini_turbos") else None,
                )
                for row in csv.DictReader(file)
            }
        counted_later = self.mini_turbo_counts()
        return {
            name: replace(lap, mini_turbos=counted_later[name])
            if lap.mini_turbos is None and name in counted_later
            else lap
            for name, lap in laps.items()
        }

    def save_mini_turbo_count(self, version_file: Path, mini_turbos: int) -> None:
        """Record the mini-turbos of a version's test lap, counted after the lap tester ran it."""
        is_new_file = not self._mini_turbo_counts.exists()
        with open(self._mini_turbo_counts, "a", newline="") as file:
            writer = csv.writer(file)
            if is_new_file:
                writer.writerow(["file", "mini_turbos"])
            writer.writerow([version_file.name, mini_turbos])

    def mini_turbo_counts(self) -> dict[str, int]:
        """Mini-turbos counted afterwards, by file name."""
        if not self._mini_turbo_counts.exists():
            return {}
        with open(self._mini_turbo_counts, newline="") as file:
            return {row["file"]: int(row["mini_turbos"]) for row in csv.DictReader(file)}

    def save(self, network_state: dict[str, Any], decisions: int, minutes: float, end_progress: float) -> Path:
        """Save a version, and return its file."""
        import torch  # only saving needs it, and it's slow to load, which would slow down listing versions

        self._folder.mkdir(parents=True, exist_ok=True)
        name = f"{training_time(minutes).replace(' ', '')}_{decisions:012d}_decisions.pt"
        temporary = self._folder / (name + ".tmp")
        torch.save({"network": network_state}, temporary)
        os.replace(temporary, self._folder / name)
        is_new_index = not self._index.exists()
        with open(self._index, "a", newline="") as file:
            writer = csv.writer(file)
            if is_new_index:
                writer.writerow(FIELDS)
            writer.writerow([name, decisions, f"{minutes:.2f}", f"{end_progress:.4f}"])
        return self._folder / name

    def all(self) -> list[Version]:
        """Every saved version, oldest first."""
        if not self._index.exists():
            return []
        with open(self._index, newline="") as file:
            return [
                Version(
                    self._folder / row["file"],
                    int(row["decisions"]),
                    float(row["minutes"]),
                    float(row["average_end_progress"]),
                )
                for row in csv.DictReader(file)
            ]

    def closest_to(self, minutes: float) -> Version:
        """The version saved nearest to this many minutes of training."""
        versions = self.all()
        if not versions:
            raise ValueError(f"no saved versions in {self._folder}")
        return min(versions, key=lambda version: abs(version.minutes - minutes))
