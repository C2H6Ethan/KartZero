"""What the reward must do. The progress changes per decision come from our real test drives."""

import pytest

from kartzero.reward import Reward

DECISIONS_PER_SECOND = 15


def race_total(reward, seconds):
    """The total score for a whole race driven at a steady pace."""
    decisions = round(seconds * DECISIONS_PER_SECOND)
    step = 1 / decisions
    return sum(reward.score(index * step, (index + 1) * step) for index in range(decisions))


@pytest.fixture
def reward():
    return Reward()


def test_being_stuck_only_costs_time(reward):
    assert reward.score(0.5, 0.5) == pytest.approx(-0.25)


def test_driving_at_the_test_drivers_pace_earns_points(reward):
    assert reward.score(0.5, 0.50052) == pytest.approx(0.79)


def test_driving_at_mario_karts_pace_earns_more(reward):
    assert reward.score(0.5, 0.50076) == pytest.approx(1.27)


def test_reversing_is_the_worst_thing_to_do(reward):
    assert reward.score(0.5, 0.49974) == pytest.approx(-0.77)


def test_progress_behind_the_start_line_works_the_same(reward):
    assert reward.score(-0.0027, -0.0022) == pytest.approx(0.75)


def test_faster_race_scores_more(reward):
    assert race_total(reward, 87.832) > race_total(reward, 128.2)


def test_reversing_then_driving_back_never_pays(reward):
    total = reward.score(0.5, 0.49) + reward.score(0.49, 0.5)
    assert total == pytest.approx(-0.5)


def test_progress_past_the_finish_pays_nothing(reward):
    # The game drives the kart by itself after the finish, and progress keeps rising.
    assert reward.score(0.9999, 1.1096) == pytest.approx(reward.score(0.9999, 1.0))


def test_the_decision_that_crosses_the_line_still_scores(reward):
    # 0.0005 of real progress up to the line, and the part past 1.0 doesn't count.
    assert reward.score(0.9995, 1.0002) == pytest.approx(0.75)


def test_scoring_after_the_finish_is_a_bug_and_fails_loudly(reward):
    # The run ends at the line, so this can only happen if that code is broken.
    with pytest.raises(ValueError):
        reward.score(1.0002, 1.0100)
