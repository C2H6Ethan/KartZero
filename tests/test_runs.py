"""Tests for finding training runs and summing them up, using made-up run folders."""

import json

import pytest
import torch

from kartzero.runs import all_runs, find_run
from kartzero.versions import LapResult, VersionStore


def make_run(runs_folder, name, settings=None, laps=()):
    """A run folder with a settings file and one version per lap, at 0, 30, 60... minutes."""
    folder = runs_folder / name
    folder.mkdir(parents=True)
    (folder / "settings.json").write_text(json.dumps(settings or {}))
    store = VersionStore(folder)
    for index, lap in enumerate(laps):
        file = store.save({"weight": torch.tensor(1.0)}, index * 25_000, index * 30.0, 1.0)
        if lap is not None:
            store.save_test_lap(file, lap)
    return folder


def test_runs_come_oldest_first(tmp_path):
    make_run(tmp_path, "20261103-210158-sticky0.5")
    make_run(tmp_path, "20261102-213453")
    assert [run.name for run in all_runs(tmp_path)] == ["20261102-213453", "20261103-210158-sticky0.5"]


def test_old_runs_without_a_sticky_setting_count_as_zero(tmp_path):
    make_run(tmp_path, "old")
    assert all_runs(tmp_path)[0].sticky_chance() == 0.0


def test_a_run_knows_where_it_started(tmp_path):
    make_run(tmp_path, "fresh")
    make_run(tmp_path, "copy", {"start_from": "runs/20261102-213453/versions/0021h44m_000031025152_decisions.pt"})
    fresh, copy = find_run("fresh", tmp_path), find_run("copy", tmp_path)
    assert fresh.started_from() == "new"
    assert copy.started_from() == "20261102-213453 at 0021h44m"


def test_the_best_lap_is_the_fastest_finish(tmp_path):
    make_run(tmp_path, "run", laps=[None, LapResult(True, 101.0, 1.0), LapResult(True, 99.5, 1.0)])
    version, lap = find_run("run", tmp_path).best_test_lap()
    assert lap.seconds == 99.5 and version.minutes == 60.0


def test_the_starting_version_never_counts_as_the_best(tmp_path):
    # Its lap belongs to the brain it was copied from, or to an AI that hasn't learned yet.
    make_run(tmp_path, "run", laps=[LapResult(True, 90.0, 1.0), LapResult(True, 101.0, 1.0)])
    _, lap = find_run("run", tmp_path).best_test_lap()
    assert lap.seconds == 101.0


def test_a_finish_beats_any_unfinished_lap(tmp_path):
    make_run(tmp_path, "run", laps=[None, LapResult(False, 200.0, 0.9), LapResult(True, 150.0, 1.0)])
    _, lap = find_run("run", tmp_path).best_test_lap()
    assert lap.finished


def test_a_run_with_no_laps_has_no_best(tmp_path):
    make_run(tmp_path, "run")
    assert find_run("run", tmp_path).best_test_lap() is None


def test_a_run_can_be_found_by_the_end_of_its_name(tmp_path):
    for name in ["20261103-210158-sticky0", "20261103-210158-sticky0.25", "20261103-210158-sticky0.5"]:
        make_run(tmp_path, name)
    assert find_run("sticky0", tmp_path).name == "20261103-210158-sticky0"
    assert find_run("0.25", tmp_path).name == "20261103-210158-sticky0.25"


def test_a_name_matching_several_runs_says_which(tmp_path):
    make_run(tmp_path, "20261103-a")
    make_run(tmp_path, "20261103-b")
    with pytest.raises(ValueError, match="20261103-a"):
        find_run("20261103", tmp_path)


def test_an_unknown_name_says_so(tmp_path):
    make_run(tmp_path, "run")
    with pytest.raises(ValueError, match="no run"):
        find_run("missing", tmp_path)


def test_a_run_lists_only_the_learning_settings_it_changed(tmp_path):
    make_run(tmp_path, "run", {
        "advantage_blend": 0.95, "exploration_weight": 0.03, "sticky_chance": 0.0,
        "emulators": 2, "decisions_per_chunk": 1024, "start_from": "x", "code_version": "abc",
    })
    assert find_run("run", tmp_path).changed_settings() == "explore 0.03, blend 0.95"


def test_a_run_with_default_settings_says_so(tmp_path):
    make_run(tmp_path, "run")
    assert find_run("run", tmp_path).changed_settings() == "defaults"
