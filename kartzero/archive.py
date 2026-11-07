"""An archive of rarely seen moments, to start some practice races from (an idea from Go-Explore).

Practice normally always starts at the start line, so a moment the AI almost never reaches stays
almost never reached. The archive remembers such moments as saved game states. Some practice
races then start from one of them, picked with more weight the rarer it is, so the AI can carry on
from there. The reward doesn't change at all: these races still score only progress and time.

"Rare" uses only the kart's own inputs (the first KART_INPUTS numbers the AI receives) and the
checkpoint, each rounded the same way, so no technique is singled out.
"""

import math
import random
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

KART_INPUTS = 11  # speed, drift angle, turning rate, mini-turbo count, boost, off-road, ground, 4 buttons
STEPS_PER_UNIT = 4  # each input is rounded to the nearest quarter of its usual range

Cell = tuple[int, ...]


def cell_of(observation: np.ndarray, checkpoint: int) -> Cell:
    """The coarse description of a moment that the archive tells moments apart by."""
    rounded = np.rint(observation[:KART_INPUTS] * STEPS_PER_UNIT).astype(int)
    return (checkpoint, *(int(value) for value in rounded))


class StateArchive:
    """Counts how often each cell is seen, and keeps saved game states for the rarest ones.

    Saving a state takes about 0.2 s, so it saves at most once every `decisions_between_saves`.
    """

    def __init__(self, folder: Path, capacity: int, decisions_between_saves: int, seed: int = 0) -> None:
        self._folder = folder
        self._capacity = capacity
        self._decisions_between_saves = decisions_between_saves
        self._random = random.Random(seed)
        self._visits: dict[Cell, int] = {}
        self._saved: dict[Cell, tuple[Path, int]] = {}  # the state, and the action held when it was saved
        self._decisions_since_save = decisions_between_saves
        self._next_file = 0
        folder.mkdir(parents=True, exist_ok=True)

    def visit(self, cell: Cell) -> bool:
        """Count one visit. Returns whether this moment should be saved now."""
        self._visits[cell] = self._visits.get(cell, 0) + 1
        self._decisions_since_save += 1
        if cell in self._saved or self._decisions_since_save < self._decisions_between_saves:
            return False
        if len(self._saved) < self._capacity:
            return True
        return self._visits[cell] < max(self._visits[c] for c in self._saved)

    def file_for(self, cell: Cell, action: int) -> Path:
        """Where to save this cell's state. Makes room by dropping the most visited saved cell."""
        self._decisions_since_save = 0
        if len(self._saved) >= self._capacity:
            commonest = max(self._saved, key=lambda c: self._visits[c])
            self._saved.pop(commonest)[0].unlink(missing_ok=True)
        file = self._folder / f"state{self._next_file}.dst"
        self._next_file += 1
        self._saved[cell] = (file, action)
        return file

    def pick(self) -> Optional[tuple[Path, int]]:
        """A saved state, picked with weight 1 / sqrt(visits), so rarer cells come up more often."""
        if not self._saved:
            return None
        cells = list(self._saved)
        weights = [1 / math.sqrt(self._visits[c]) for c in cells]
        chosen = self._random.choices(cells, weights)[0]
        self._visits[chosen] += 1  # being picked counts, so one cell can't be picked forever
        return self._saved[chosen]

    def size(self) -> int:
        return len(self._saved)

    def cells_seen(self) -> int:
        return len(self._visits)

    def delete(self) -> None:
        shutil.rmtree(self._folder, ignore_errors=True)
