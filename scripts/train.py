"""Train the AI on Figure-8. Closing the viewer window stops training and saves its progress.

Usage: python scripts/train.py [--resume] [--minutes N] [--no-viewer] [--emulators N]
Run make_start_state.py first.
"""

import argparse
import multiprocessing as mp


def show_viewer() -> None:
    from kartzero.viewer import run_viewer

    run_viewer()


def main() -> None:
    # PyTorch is imported here, so that the emulator processes don't each load it.
    from kartzero.learner import Settings, run_training

    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=None)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--emulators", type=int, default=8)
    parser.add_argument("--resume", action="store_true", help="carry on the last run from checkpoints/latest.pt")
    arguments = parser.parse_args()

    viewer = None
    if not arguments.no_viewer:
        viewer = mp.get_context("spawn").Process(target=show_viewer)
        viewer.start()
    run_training(Settings(emulators=arguments.emulators), arguments.minutes, viewer, arguments.resume)
    if viewer is not None and viewer.is_alive():
        viewer.terminate()


if __name__ == "__main__":
    main()
