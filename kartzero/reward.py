"""The AI's score for each decision: points for track progress, minus points for time."""

POINTS_PER_PROGRESS = 2000
TIME_PENALTY = 0.25
from kartzero.kart import FINISH_PROGRESS


class Reward:
    """Scores one decision at a time. It only uses what it's given, so it keeps no state."""

    def score(self, progress_before: float, progress_now: float) -> float:
        """The reward for one decision, given race_progress before and after it."""
        if progress_before >= FINISH_PROGRESS:
            raise ValueError("progress_before must be below " + str(FINISH_PROGRESS))
        
        progress_now = min(FINISH_PROGRESS, progress_now)

        delta = progress_now - progress_before

        return POINTS_PER_PROGRESS * delta - TIME_PENALTY
