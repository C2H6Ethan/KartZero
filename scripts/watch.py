"""Watch the AI race in a window, without training, or list the runs and their test laps.

Usage:
  python scripts/watch.py                          the newest AI, from checkpoints/latest.pt
  python scripts/watch.py --list                   every run: its settings, best and newest test lap
  python scripts/watch.py --list sticky0.5         every saved version of one run, with its test lap
  python scripts/watch.py --run sticky0.5          one run, always its newest saved version: each race uses the newest
  python scripts/watch.py --hours 30               the version nearest 30 hours of training, in the newest run
  python scripts/watch.py --hours 1 --run 0.25     the same, in one run
  python scripts/watch.py --hours 5.9 --run start-practice --test-lap
                                                   that version's exact test lap: the likeliest buttons, no dice
  python scripts/watch.py --compare                take turns between the sticky comparison runs, one race each

A run can be named by its whole folder name, or the end of it, like 213453 or sticky0.25.
"""

import argparse
import math
from pathlib import Path
from typing import Callable, Optional

from kartzero.runs import TrainingRun, all_runs, find_run
from kartzero.versions import LapResult, training_time

ROOT = Path(__file__).resolve().parent.parent
ALL_RUNS = "*"


def watch(brain: Optional[Callable[[], Path]] = None, test_lap: bool = False) -> None:
    """Race over and over. `brain` says which saved AI to drive, asked again before every race."""
    # Loaded only here, because PyTorch and the emulator are slow to load and --list needs neither.
    from kartzero.viewer import LATEST_CHECKPOINT, run_viewer

    run_viewer(brain or (lambda: LATEST_CHECKPOINT), test_lap)


def newest_version(run: TrainingRun) -> Path:
    """The run's newest saved version right now. Training keeps adding versions, so ask again each race."""
    return run.versions.all()[-1].file


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
    starts = run.start_practice_by_version()
    if starts:
        # In the 75-decision races of the countdown setup: 0.0198 is the full start boost, 0.0051 a
        # plain start at GO, and about 0 a stall. Practice rolls the dice, so it stays below the best.
        print("start practice: progress 2 s after GO, 0.0198 = full start boost, 0.0051 = plain start\n")
    print("training time    decisions   practice progress   test lap   mini-turbos" + ("   start practice" if starts else ""))
    for version in run.versions.all():
        progress = "-" if math.isnan(version.average_end_progress) else f"{version.average_end_progress:.1%}"
        marks = "  best" if best and best[0].file == version.file else ""
        marks += "  start" if version.decisions == 0 else ""
        lap = test_laps.get(version.file.name)
        mini_turbos = "-" if lap is None or lap.mini_turbos is None else str(lap.mini_turbos)
        start = f"{starts[version.file.name]:.4f}" if version.file.name in starts else "-"
        start_column = f"   {start:>14}" if starts else ""
        print(
            f"{training_time(version.minutes)}  {version.decisions:11,}   {progress:>17}   "
            f"{lap_text(lap):>8}   {mini_turbos:>11}{start_column}{marks}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list", nargs="?", const=ALL_RUNS, metavar="RUN", help="list every run, or every version of one run"
    )
    parser.add_argument("--run", help="which run to watch, for --hours or on its own")
    parser.add_argument("--hours", type=float, help="watch the version nearest this many hours of training")
    parser.add_argument("--test-lap", action="store_true", help="drive like a test lap: the likeliest buttons, no dice")
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
            if arguments.hours is None:
                print(f"watching {run.name}, each race with its newest saved version")
                watch(lambda: newest_version(run), arguments.test_lap)
            else:
                version = run.versions.closest_to(arguments.hours * 60)
                print(f"watching {run.name} after {training_time(version.minutes)} of training, {version.decisions:,} decisions")
                watch(lambda: version.file, arguments.test_lap)
        else:
            watch(test_lap=arguments.test_lap)
    except ValueError as error:
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()
