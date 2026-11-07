"""Finds the training runs in runs/ and sums each one up, without loading PyTorch or the emulator."""

import json
from pathlib import Path
from typing import Any, Optional

from kartzero.settings import DEFAULTS
from kartzero.versions import LapResult, Version, VersionStore

RUNS_FOLDER = Path(__file__).resolve().parent.parent / "runs"
# Shown in their own column, or not about what is learned: emulators x decisions per chunk is what matters.
NOT_LISTED = {"start_from", "emulators", "decisions_per_chunk"}
SHORT_NAMES = {"sticky_chance": "sticky", "exploration_weight": "explore", "advantage_blend": "blend"}


class TrainingRun:
    """One folder in runs/: its settings, saved versions and test laps."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.name = folder.name
        self.versions = VersionStore(folder)

    def settings(self) -> dict[str, Any]:
        file = self.folder / "settings.json"
        settings: dict[str, Any] = json.loads(file.read_text()) if file.exists() else {}
        return settings

    def sticky_chance(self) -> float:
        return float(self.settings().get("sticky_chance", 0.0))  # runs from before stickiness had none

    def changed_settings(self) -> str:
        """The learning settings that differ from the defaults, like "explore 0.03, blend 0.95"."""
        settings = self.settings()
        changed = [
            f"{SHORT_NAMES.get(name, name)} {settings[name]:g}" if isinstance(settings[name], float) else f"{name} {settings[name]}"
            for name, default in DEFAULTS.items()
            if name in settings and name not in NOT_LISTED and settings[name] != default
        ]
        return ", ".join(changed) or "defaults"

    def started_from(self) -> str:
        """ "new" for an untrained start, otherwise the run and training time of the brain it copied."""
        start = self.settings().get("start_from", "")
        if not start:
            return "new"
        file = Path(start)
        return f"{file.parent.parent.name} at {file.name.split('_')[0]}"

    def test_laps(self) -> list[tuple[Version, LapResult]]:
        """Every version that has a test lap, oldest first."""
        laps = self.versions.test_laps()
        return [(version, laps[version.file.name]) for version in self.versions.all() if version.file.name in laps]

    def best_test_lap(self) -> Optional[tuple[Version, LapResult]]:
        """The fastest finish, or the furthest lap if none finished. The starting version doesn't count,
        because its lap belongs to the brain it was copied from, or to an AI that hasn't learned yet."""
        laps = [(version, lap) for version, lap in self.test_laps() if version.decisions > 0]
        if not laps:
            return None
        return min(laps, key=lambda pair: (not pair[1].finished, pair[1].seconds, -pair[1].progress))


def all_runs(runs_folder: Path = RUNS_FOLDER) -> list[TrainingRun]:
    """Every run, oldest first. Folder names start with the date and time, so name order is time order."""
    if not runs_folder.exists():
        return []
    return [TrainingRun(folder) for folder in sorted(runs_folder.iterdir()) if folder.is_dir()]


def find_run(text: str, runs_folder: Path = RUNS_FOLDER) -> TrainingRun:
    """The run with this name, or else the one whose name ends with it, or else the one containing it."""
    runs = all_runs(runs_folder)
    for matches in (
        [run for run in runs if run.name == text],
        [run for run in runs if run.name.endswith(text)],
        [run for run in runs if text in run.name],
    ):
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"'{text}' matches several runs: {', '.join(run.name for run in matches)}")
    raise ValueError(f"no run matches '{text}'")
