"""Keeps past versions of the AI to watch later, without ending up with thousands of them.

Versions are close together early on, when the AI changes fastest, and further apart later:
each one waits 3% longer than the last, and at least about a minute. That gives about 200
versions by 100 million decisions.
"""

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

SMALLEST_GAP = 25_000  # about a minute of training
GROWTH_PER_VERSION = 1.03
INDEX_FILE = "versions.csv"
FIELDS = ["file", "decisions", "minutes", "average_end_progress"]


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


class VersionStore:
    """The saved versions of one training run, in its folder, with an index you can open in Excel."""

    def __init__(self, run_folder: Path) -> None:
        self._folder = run_folder / "versions"
        self._index = self._folder / INDEX_FILE

    def save(self, network_state: dict[str, Any], decisions: int, minutes: float, end_progress: float) -> None:
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
