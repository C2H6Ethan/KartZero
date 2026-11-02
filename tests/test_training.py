"""What the learning maths must do."""

import pytest
import torch

from kartzero.training import advantages, future_points, guess_error, limited_scores


def test_a_perfect_guess_has_no_error():
    assert guess_error(torch.tensor([300.0]), torch.tensor([300.0])).item() == 0.0


def test_the_error_is_the_difference_squared():
    assert guess_error(torch.tensor([100.0]), torch.tensor([60.0])).item() == pytest.approx(1600.0)


def test_guessing_too_low_counts_the_same_as_too_high():
    too_low = guess_error(torch.tensor([20.0]), torch.tensor([60.0]))
    too_high = guess_error(torch.tensor([100.0]), torch.tensor([60.0]))
    assert too_low.item() == pytest.approx(too_high.item())


def test_several_guesses_give_one_average():
    # Errors of 2 and 4 become 4 and 16, which average 10.
    error = guess_error(torch.tensor([12.0, 24.0]), torch.tensor([10.0, 20.0]))
    assert error.item() == pytest.approx(10.0)


def test_training_can_use_it_to_pull_a_guess_towards_the_real_points():
    guess = torch.tensor([100.0], requires_grad=True)
    guess_error(guess, torch.tensor([60.0])).backward()
    assert guess.grad.item() > 0  # a positive slope means lowering the guess lowers the error


def score_of(ratio, advantage):
    return limited_scores(torch.tensor([ratio]), torch.tensor([advantage])).item()


def push_on_odds(ratio, advantage):
    """How strongly training would push the ratio: above 0 is up, below 0 is down, 0 is no push."""
    ratios = torch.tensor([ratio], requires_grad=True)
    limited_scores(ratios, torch.tensor([advantage])).sum().backward()
    return ratios.grad.item()


def test_inside_the_limit_the_score_is_ratio_times_advantage():
    assert score_of(1.1, 50.0) == pytest.approx(55.0)


def test_above_the_limit_the_score_stops_growing():
    assert score_of(1.3, 50.0) == pytest.approx(60.0)


def test_good_button_inside_the_limit_is_pushed_up():
    assert push_on_odds(1.1, 50.0) == pytest.approx(50.0)


def test_good_button_past_the_limit_is_no_longer_pushed():
    assert push_on_odds(1.3, 50.0) == 0.0


def test_good_button_that_fell_too_far_is_still_pushed_back_up():
    assert push_on_odds(0.7, 50.0) == pytest.approx(50.0)


def test_bad_button_inside_the_limit_is_pushed_down():
    assert push_on_odds(0.9, -50.0) == pytest.approx(-50.0)


def test_bad_button_past_the_limit_is_no_longer_pushed():
    assert push_on_odds(0.7, -50.0) == 0.0


def test_bad_button_that_rose_too_far_is_still_pushed_back_down():
    assert push_on_odds(1.3, -50.0) == pytest.approx(-50.0)


def test_each_decision_gets_its_own_score():
    scores = limited_scores(torch.tensor([1.1, 1.3]), torch.tensor([50.0, 50.0]))
    assert scores.tolist() == pytest.approx([55.0, 60.0])


def test_better_than_expected_is_positive():
    assert advantages([350.0], [300.0]) == pytest.approx([50.0])


def test_worse_than_expected_is_negative():
    assert advantages([60.0], [100.0]) == pytest.approx([-40.0])


def test_exactly_as_expected_is_zero():
    assert advantages([300.0], [300.0]) == pytest.approx([0.0])


def test_each_decision_is_compared_with_its_own_guess():
    assert advantages([10.0, 20.0, 30.0], [1.0, 2.0, 3.0]) == pytest.approx([9.0, 18.0, 27.0])


def test_lists_of_different_lengths_are_a_bug_and_fail_loudly():
    with pytest.raises(ValueError):
        advantages([1.0, 2.0], [1.0])


def test_the_worked_example():
    assert future_points([1, 1, -0.25], gamma=0.5) == pytest.approx([1.4375, 0.875, -0.25])


def test_a_single_decision_is_just_its_reward():
    assert future_points([3], gamma=0.5) == pytest.approx([3])


def test_a_gamma_of_one_adds_everything_up():
    assert future_points([1, 2, 3], gamma=1.0) == pytest.approx([6, 5, 3])


def test_a_gamma_of_zero_only_counts_the_decision_itself():
    assert future_points([1, 2, 3], gamma=0.0) == pytest.approx([1, 2, 3])


def test_a_recording_that_stops_mid_race_includes_what_comes_after():
    assert future_points([1, 1], gamma=0.5, points_after=4) == pytest.approx([2.5, 3])


def test_the_input_list_is_left_unchanged():
    rewards = [1, 1, -0.25]
    future_points(rewards, gamma=0.5)
    assert rewards == [1, 1, -0.25]


def test_an_empty_recording_gives_an_empty_answer():
    assert future_points([], gamma=0.5) == []
