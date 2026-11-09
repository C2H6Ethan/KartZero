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
    hold_lengths: bool = False  # the AI also picks how long to hold the pedal and the steering, see holds.py
    curiosity_weight: float = 0.0  # a bonus for reaching moments it rarely sees, 0 is off, see curiosity.py
    curiosity_fade_decisions: int = 0  # lower the curiosity weight to 0 over this many decisions, 0 is never
    curiosity_learning_rate: float = 1e-4  # how fast moments become familiar: slower keeps them new for longer
    # Practice only: this many emulators drive short races that end soon after GO, so the AI gets far
    # more tries at the start. The rest drive full races, so it keeps practising the whole track.
    start_practice_emulators: int = 0
    start_practice_decisions: int = 75  # 44 in the countdown, then about 2 s of racing
    start_from: str = ""  # a saved version to start from, instead of an untrained network


DEFAULTS = {field.name: field.default for field in fields(Settings)}
