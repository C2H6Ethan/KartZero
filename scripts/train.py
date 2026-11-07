"""Train the AI on Figure-8. Closing the viewer window stops training and saves its progress.

Usage: python scripts/train.py [--resume] [--minutes N] [--no-viewer] [--emulators N]
    [--decisions-per-chunk N] [--sticky-chance P] [--sticky-per-button] [--archive-start-chance P] [--self-imitation-weight W] [--exploration-weight W] [--advantage-blend L]
    [--start-from VERSION] [--checkpoint FILE] [--name NAME]
Run make_start_state.py first.
"""

import argparse
import multiprocessing as mp
from pathlib import Path


def show_viewer() -> None:
    from kartzero.viewer import run_viewer

    run_viewer()


def main() -> None:
    # PyTorch is imported here, so that the emulator processes don't each load it.
    from kartzero.learner import LATEST_CHECKPOINT, Settings, run_training

    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=None)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--emulators", type=int, default=8)
    parser.add_argument("--decisions-per-chunk", type=int, default=256)
    parser.add_argument("--sticky-chance", type=float, default=0.0, help="practice only: chance to repeat the last button")
    parser.add_argument("--sticky-per-button", action="store_true", help="separate sticky dice for pedal and steering")
    parser.add_argument("--hold-lengths", action="store_true", help="the AI also picks how long to hold each half")
    parser.add_argument("--archive-start-chance", type=float, default=0.0, help="chance a practice race starts from a rare saved moment")
    parser.add_argument("--archive-explore-decisions", type=int, default=30, help="random buttons for up to this many decisions after an archive start")
    parser.add_argument("--curiosity-weight", type=float, default=0.0, help="bonus for reaching new moments, 0 is off")
    parser.add_argument("--self-imitation-weight", type=float, default=0.0, help="copy past decisions that beat the guess, 0 is off")
    parser.add_argument("--exploration-weight", type=float, default=0.01, help="how much to reward keeping options open")
    parser.add_argument("--advantage-blend", type=float, default=1.0, help="GAE's lambda: 1 is off, 0.95 is common")
    parser.add_argument("--start-from", default="", help="a saved version file to start from")
    parser.add_argument("--checkpoint", type=Path, default=LATEST_CHECKPOINT, help="where to save, and resume from")
    parser.add_argument("--name", default="", help="added to the run folder's name")
    parser.add_argument("--resume", action="store_true", help="carry on the run saved in --checkpoint")
    arguments = parser.parse_args()

    viewer = None
    if not arguments.no_viewer:
        viewer = mp.get_context("spawn").Process(target=show_viewer)
        viewer.start()
    settings = Settings(
        emulators=arguments.emulators,
        decisions_per_chunk=arguments.decisions_per_chunk,
        sticky_chance=arguments.sticky_chance,
        sticky_per_button=arguments.sticky_per_button,
        hold_lengths=arguments.hold_lengths,
        archive_start_chance=arguments.archive_start_chance,
        archive_explore_decisions=arguments.archive_explore_decisions,
        self_imitation_weight=arguments.self_imitation_weight,
        curiosity_weight=arguments.curiosity_weight,
        exploration_weight=arguments.exploration_weight,
        advantage_blend=arguments.advantage_blend,
        start_from=arguments.start_from,
    )
    run_training(settings, arguments.minutes, viewer, arguments.resume, arguments.checkpoint, arguments.name)
    if viewer is not None and viewer.is_alive():
        viewer.terminate()


if __name__ == "__main__":
    main()
