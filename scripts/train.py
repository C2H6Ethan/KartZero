"""Train the AI on Figure-8. Closing the viewer window stops training and saves its progress.

Usage: python scripts/train.py [--resume] [--minutes N] [--seed N] [--no-viewer] [--emulators N]
    [--decisions-per-chunk N] [--sticky-chance P] [--hold-lengths]
    [--curiosity-weight W] [--curiosity-fade-decisions N] [--curiosity-learning-rate R] [--exploration-weight W] [--advantage-blend L] [--start-from VERSION] [--checkpoint FILE] [--name NAME]
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
    parser.add_argument("--seed", type=int, default=1, help="starts every random number in the run, so the same seed repeats it")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--emulators", type=int, default=8)
    parser.add_argument("--decisions-per-chunk", type=int, default=256)
    parser.add_argument("--sticky-chance", type=float, default=0.0, help="practice only: chance to repeat the last button")
    parser.add_argument("--hold-lengths", action="store_true", help="the AI also picks how long to hold each half")
    parser.add_argument("--curiosity-weight", type=float, default=0.0, help="bonus for reaching new moments, 0 is off")
    parser.add_argument("--curiosity-fade-decisions", type=int, default=0, help="fade curiosity to 0 over this many decisions")
    parser.add_argument("--curiosity-learning-rate", type=float, default=1e-4, help="slower keeps moments new for longer")
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
        seed=arguments.seed,
        emulators=arguments.emulators,
        decisions_per_chunk=arguments.decisions_per_chunk,
        sticky_chance=arguments.sticky_chance,
        hold_lengths=arguments.hold_lengths,
        curiosity_weight=arguments.curiosity_weight,
        curiosity_fade_decisions=arguments.curiosity_fade_decisions,
        curiosity_learning_rate=arguments.curiosity_learning_rate,
        exploration_weight=arguments.exploration_weight,
        advantage_blend=arguments.advantage_blend,
        start_from=arguments.start_from,
    )
    run_training(settings, arguments.minutes, viewer, arguments.resume, arguments.checkpoint, arguments.name)
    if viewer is not None and viewer.is_alive():
        viewer.terminate()


if __name__ == "__main__":
    main()
