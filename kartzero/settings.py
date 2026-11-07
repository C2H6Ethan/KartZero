"""Everything that shapes a training run. Kept apart from the learner, so reading it doesn't load PyTorch."""

from dataclasses import dataclass, fields


@dataclass
class Settings:
    """Everything that shapes a run. Saved with its results, so runs can be reproduced."""

    seed: int = 1
    emulators: int = 8
    decisions_per_chunk: int = 256
    gamma: float = 0.99
    learning_rate: float = 3e-4
    learning_passes: int = 4
    batch_size: int = 512
    guess_weight: float = 0.5
    exploration_weight: float = 0.01
    advantage_blend: float = 1.0  # GAE's lambda: 1 is the real points minus the guess, lower leans on guesses
    sticky_chance: float = 0.0  # practice only: the chance each decision repeats the last button
    sticky_per_button: bool = False
    hold_lengths: bool = False  # the AI also picks how long to hold the pedal and the steering (FiGAR)  # roll the sticky dice for the pedal and the steering separately
    # Go-Explore style starts: the chance a practice race starts from an archived rare moment.
    archive_start_chance: float = 0.0
    archive_capacity: int = 100  # saved states kept per emulator, about 2.7 MB each
    archive_race_decisions: int = 300  # races from the archive stop after this many decisions
    archive_explore_decisions: int = 30  # random buttons for up to this many decisions after an archive start
    archive_decisions_between_saves: int = 500  # saving takes about 0.2 s, so ration it
    # Self-imitation: 0 is off. Otherwise how strongly to copy past decisions that beat the guess.
    self_imitation_weight: float = 0.0
    self_imitation_buffer: int = 200_000  # decisions kept, about 12 minutes of practice at 1M decisions per hour
    self_imitation_batches: int = 4  # extra learning steps per round
    curiosity_weight: float = 0.0  # 0 is off: a bonus for reaching new moments, see curiosity.py
    start_from: str = ""  # a saved version to start from, instead of an untrained network


DEFAULTS = {field.name: field.default for field in fields(Settings)}
