"""Watch the AI race in a window, without training. Close the window to stop.

Usage:
  python scripts/watch.py              the newest AI
  python scripts/watch.py --list       every saved version of the latest run
  python scripts/watch.py --hours 30   the AI as it was after about 30 hours of training
"""

import argparse
import math

from kartzero.game import ROOT
from kartzero.versions import VersionStore, training_time
from kartzero.viewer import run_viewer


def latest_store() -> VersionStore:
    runs = sorted(folder for folder in (ROOT / "runs").iterdir() if (folder / "versions").exists())
    if not runs:
        raise SystemExit("no saved versions yet: train first")
    return VersionStore(runs[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="show every saved version")
    parser.add_argument("--hours", type=float, help="watch the version nearest this many hours of training")
    arguments = parser.parse_args()

    if arguments.list:
        print("training time    decisions   race progress")
        for version in latest_store().all():
            progress = "-" if math.isnan(version.average_end_progress) else f"{version.average_end_progress:.1%}"
            print(f"{training_time(version.minutes)}  {version.decisions:11,}   {progress:>6}")
    elif arguments.hours is not None:
        version = latest_store().closest_to(arguments.hours * 60)
        print(f"watching the AI after {training_time(version.minutes)} of training, {version.decisions:,} decisions")
        run_viewer(version.file)
    else:
        run_viewer()


if __name__ == "__main__":
    main()
