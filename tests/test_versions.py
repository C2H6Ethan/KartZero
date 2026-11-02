"""Tests for how often versions are saved, and for finding them again."""

import pytest
import torch

from kartzero.versions import SMALLEST_GAP, VersionStore, next_version_after, training_time


def test_training_time_shows_hours_and_minutes():
    assert training_time(30 * 60 + 12.9) == "0030h 12m"
    assert training_time(0.0) == "0000h 00m"


def test_a_version_file_name_shows_its_training_time(tmp_path):
    store = VersionStore(tmp_path)
    store.save({"weight": torch.tensor(1.0)}, 5_000_000, 30 * 60 + 12.0, 0.5)
    assert store.all()[0].file.name.startswith("0030h12m_")


def test_the_first_version_comes_early():
    assert next_version_after(0) == SMALLEST_GAP


def test_versions_are_never_closer_than_the_smallest_gap():
    decisions = 0
    while decisions < 5_000_000:
        following = next_version_after(decisions)
        assert following - decisions >= SMALLEST_GAP
        decisions = following


def test_the_next_version_is_always_ahead():
    for decisions in (0, 20_000, 1_000_000, 50_000_000):
        assert next_version_after(decisions) > decisions


def versions_between(start, end):
    decisions, count = start, 0
    while decisions < end:
        decisions = next_version_after(decisions)
        count += 1
    return count


def test_versions_get_further_apart_as_training_goes_on():
    assert versions_between(10_000_000, 11_000_000) < versions_between(0, 1_000_000) / 5


def test_a_long_training_run_stays_in_the_hundreds():
    decisions, count = 0, 0
    while decisions < 100_000_000:
        decisions = next_version_after(decisions)
        count += 1
    assert 100 <= count < 400


@pytest.fixture
def store(tmp_path):
    store = VersionStore(tmp_path)
    store.save({"weight": torch.tensor(1.0)}, 20_000, 0.5, 0.1)
    store.save({"weight": torch.tensor(2.0)}, 1_000_000, 30.0, 0.8)
    return store


def test_versions_come_back_in_order_with_their_details(store):
    versions = store.all()
    assert [version.decisions for version in versions] == [20_000, 1_000_000]
    assert versions[1].minutes == 30.0
    assert versions[1].average_end_progress == 0.8


def test_a_saved_version_holds_that_network(store):
    saved = torch.load(store.all()[0].file)
    assert saved["network"]["weight"].item() == 1.0


def test_the_closest_version_is_found_by_minutes(store):
    assert store.closest_to(25.0).decisions == 1_000_000
    assert store.closest_to(1.0).decisions == 20_000


def test_an_empty_store_has_no_versions(tmp_path):
    assert VersionStore(tmp_path).all() == []
    with pytest.raises(ValueError):
        VersionStore(tmp_path).closest_to(5.0)
