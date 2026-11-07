"""Watch the AI race in a window, without training, or list the runs and their test laps.

Usage:
  python scripts/watch.py                          the newest AI, from checkpoints/latest.pt
  python scripts/watch.py --list                   every run: its settings, best and newest test lap
  python scripts/watch.py --list sticky0.5         every saved version of one run, with its test lap
  python scripts/watch.py --run sticky0.5          the newest saved version of one run
  python scripts/watch.py --hours 30               the version nearest 30 hours of training, in the newest run
  python scripts/watch.py --hours 1 --run 0.25     the same, in one run
  python scripts/watch.py --compare                take turns between the sticky comparison runs, one race each

A run can be named by its whole folder name, or the end of it, like 213453 or sticky0.25.
"""

import argparse
import math
from pathlib import Path
from typing import Optional

from kartzero.runs import TrainingRun, all_runs, find_run
from kartzero.versions import LapResult, training_time

ROOT = Path(__file__).resolve().parent.parent
ALL_RUNS = "*"


def watch(checkpoint: Optional[Path] = None) -> None:
    # Loaded only here, because PyTorch and the emulator are slow to load and --list needs neither.
    from kartzero.viewer import LATEST_CHECKPOINT, run_viewer

    run_viewer(checkpoint or LATEST_CHECKPOINT)


def watch_comparison() -> None:
    """Each run saves to checkpoints/compare-sticky<chance>.pt, so the chance comes from the name."""
    from kartzero.viewer import ViewerRun, run_rotating_viewer

    files = sorted((ROOT / "checkpoints").glob("compare-sticky*.pt"), key=lambda f: float(f.stem[len("compare-sticky"):]))
    if not files:
        raise SystemExit("no comparison runs found in checkpoints/")
    runs = []
    for file in files:
        chance = float(file.stem[len("compare-sticky"):])
        runs.append(ViewerRun(f"sticky {chance:g}", file, chance))
    run_rotating_viewer(runs)


def lap_text(lap: Optional[LapResult]) -> str:
    return lap.description() if lap else "not yet"


def list_runs() -> None:
    runs = [run for run in all_runs() if run.versions.all()]
    if not runs:
        raise SystemExit("no saved versions yet: train first")
    width = max(len(run.name) for run in runs)
    start_width = max(len("started from"), *(len(run.started_from()) for run in runs))
    settings_width = max(len("settings"), *(len(run.changed_settings()) for run in runs))
    print(
        f"{'run':<{width}}  {'settings':<{settings_width}}  {'started from':<{start_width}}  "
        f"trained    versions  {'best test lap':<22}  newest"
    )
    for run in runs:
        versions = run.versions.all()
        best = run.best_test_lap()
        best_text = f"{best[1].description()} at {training_time(best[0].minutes)}" if best else "-"
        laps = run.test_laps()
        print(
            f"{run.name:<{width}}  {run.changed_settings():<{settings_width}}  {run.started_from():<{start_width}}  "
            f"{training_time(versions[-1].minutes)}  {len(versions):>8}  {best_text:<22}  "
            f"{lap_text(laps[-1][1] if laps else None)}"
        )
    print("\n'trained' is the newest saved version. A run started from another counts its time from 0.")
    print("The first version of a run is its starting brain, so it never counts as the best.")


def list_versions(run: TrainingRun) -> None:
    print(f"{run.name}: {run.changed_settings()}, started from {run.started_from()}\n")
    test_laps = run.versions.test_laps()
    best = run.best_test_lap()
    print("training time    decisions   practice progress   test lap")
    for version in run.versions.all():
        progress = "-" if math.isnan(version.average_end_progress) else f"{version.average_end_progress:.1%}"
        marks = "  best" if best and best[0].file == version.file else ""
        marks += "  start" if version.decisions == 0 else ""
        print(
            f"{training_time(version.minutes)}  {version.decisions:11,}   {progress:>17}   "
            f"{lap_text(test_laps.get(version.file.name)):>8}{marks}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list", nargs="?", const=ALL_RUNS, metavar="RUN", help="list every run, or every version of one run"
    )
    parser.add_argument("--run", help="which run to watch, for --hours or on its own")
    parser.add_argument("--hours", type=float, help="watch the version nearest this many hours of training")
    parser.add_argument("--compare", action="store_true", help="take turns between the sticky comparison runs")
    arguments = parser.parse_args()

    try:
        if arguments.compare:
            watch_comparison()
        elif arguments.list == ALL_RUNS:
            list_runs()
        elif arguments.list is not None:
            list_versions(find_run(arguments.list))
        elif arguments.hours is not None or arguments.run:
            runs = [find_run(arguments.run)] if arguments.run else [r for r in all_runs() if r.versions.all()]
            if not runs:
                raise SystemExit("no saved versions yet: train first")
            run = runs[-1]
            versions = run.versions
            version = versions.closest_to(arguments.hours * 60) if arguments.hours is not None else versions.all()[-1]
            print(f"watching {run.name} after {training_time(version.minutes)} of training, {version.decisions:,} decisions")
            watch(version.file)
        else:
            watch()
    except ValueError as error:
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()
