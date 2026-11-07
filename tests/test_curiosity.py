"""Tests for the curiosity bonus."""

import torch

from kartzero.curiosity import WARMUP_CHUNKS, Curiosity, faded_weight


def test_moments_seen_often_become_less_new_than_unseen_ones():
    torch.manual_seed(0)
    curiosity = Curiosity(6, torch.device("cpu"))
    familiar = torch.randn(2048, 6) * 0.1
    for _ in range(30):
        curiosity.learn(familiar)
    unseen = torch.randn(256, 6) * 0.1 + torch.tensor([0.0, 0.0, 0.0, 3.0, 0.0, 0.0])
    assert curiosity.novelty(unseen).mean() > 3 * curiosity.novelty(familiar[:256]).mean()


def warmed_up(inputs=6):
    curiosity = Curiosity(inputs, torch.device("cpu"))
    for _ in range(WARMUP_CHUNKS):
        curiosity.learn(torch.randn(64, inputs), passes=1)
    return curiosity


def test_the_bonus_averages_zero_and_scales_with_its_weight():
    curiosity = warmed_up()
    novelty = curiosity.novelty(torch.randn(100, 6))
    small, large = curiosity.bonus(novelty, 0.1), curiosity.bonus(novelty, 0.2)
    assert abs(float(small.mean())) < 1e-6
    assert torch.allclose(large, 2 * small)
    assert small.abs().sum() > 0


def test_no_bonus_is_paid_while_it_warms_up():
    curiosity = Curiosity(6, torch.device("cpu"))
    for _ in range(WARMUP_CHUNKS - 1):
        curiosity.learn(torch.randn(64, 6), passes=1)
    novelty = curiosity.novelty(torch.randn(100, 6))
    assert torch.equal(curiosity.bonus(novelty, 0.3), torch.zeros(100))


def test_a_saved_state_scores_moments_exactly_the_same(tmp_path):
    original = warmed_up()
    torch.save(original.state(), tmp_path / "curiosity.pt")
    restored = Curiosity(6, torch.device("cpu"), seed=99)  # different random networks until loaded
    restored.load_state(torch.load(tmp_path / "curiosity.pt"))
    moments = torch.randn(50, 6)
    assert torch.equal(restored.novelty(moments), original.novelty(moments))
    assert torch.equal(restored.bonus(restored.novelty(moments), 0.3), original.bonus(original.novelty(moments), 0.3))


def test_the_weight_fades_in_a_straight_line_then_stays_at_zero():
    assert faded_weight(0.3, 1000, 0) == 0.3
    assert abs(faded_weight(0.3, 1000, 500) - 0.15) < 1e-12
    assert faded_weight(0.3, 1000, 1000) == 0.0
    assert faded_weight(0.3, 1000, 5000) == 0.0
    assert faded_weight(0.3, 0, 5000) == 0.3  # 0 means no fade


def test_curiosity_does_not_change_the_ais_own_random_numbers():
    torch.manual_seed(5)
    expected = torch.rand(3)
    torch.manual_seed(5)
    Curiosity(6, torch.device("cpu"), seed=1)
    assert torch.equal(torch.rand(3), expected)


def test_a_slower_learning_rate_keeps_moments_new_for_longer():
    torch.manual_seed(0)
    familiar = torch.randn(2048, 6) * 0.1
    errors = []
    for rate in (1e-4, 2.5e-5):
        curiosity = Curiosity(6, torch.device("cpu"), learning_rate=rate)
        for _ in range(10):
            curiosity.learn(familiar)
        with torch.no_grad():
            errors.append(float(curiosity._error(familiar).mean()))
    assert errors[1] > errors[0]
