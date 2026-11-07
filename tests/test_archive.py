"""Tests for the archive of rare moments. Saving is faked by touching files."""

import numpy as np

from kartzero.archive import KART_INPUTS, StateArchive, cell_of


def test_a_cell_rounds_each_kart_input_the_same_way_and_ignores_the_rest():
    observation = np.zeros(35, dtype=np.float32)
    observation[0] = 0.77  # speed 7.7 rounds to 0.75
    observation[3] = 0.5  # mini-turbo count 2
    observation[KART_INPUTS:] = 0.9  # sensors don't count
    assert cell_of(observation, 5) == (5, 3, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0)


def save(archive, cell, action=0):
    file = archive.file_for(cell, action)
    file.touch()
    return file


def test_a_new_cell_is_saved_and_a_seen_one_is_not(tmp_path):
    archive = StateArchive(tmp_path, capacity=10, decisions_between_saves=1)
    assert archive.visit((1,))
    save(archive, (1,))
    assert not archive.visit((1,))


def test_saves_are_rationed(tmp_path):
    archive = StateArchive(tmp_path, capacity=10, decisions_between_saves=3)
    assert archive.visit((1,))
    save(archive, (1,))
    assert not archive.visit((2,))
    assert not archive.visit((3,))
    assert archive.visit((4,))


def test_a_full_archive_drops_its_most_visited_cell(tmp_path):
    archive = StateArchive(tmp_path, capacity=2, decisions_between_saves=1)
    archive.visit((1,))
    common = save(archive, (1,))
    archive.visit((2,))
    save(archive, (2,))
    for _ in range(5):
        archive.visit((1,))
    assert archive.visit((3,))
    save(archive, (3,))
    assert archive.size() == 2
    assert not common.exists()


def test_a_full_archive_does_not_save_a_cell_commoner_than_all_it_keeps(tmp_path):
    archive = StateArchive(tmp_path, capacity=1, decisions_between_saves=1)
    archive.visit((1,))
    save(archive, (1,))
    for _ in range(3):
        archive.visit((2,))
    assert not archive.visit((2,))


def test_rarer_cells_are_picked_more_often(tmp_path):
    archive = StateArchive(tmp_path, capacity=2, decisions_between_saves=1, seed=1)
    archive.visit((1,))
    rare = save(archive, (1,), action=7)
    archive.visit((2,))
    save(archive, (2,))
    for _ in range(400):
        archive.visit((2,))
    picks = [archive.pick() for _ in range(200)]
    assert sum(pick == (rare, 7) for pick in picks) > 100


def test_an_empty_archive_picks_nothing(tmp_path):
    assert StateArchive(tmp_path, capacity=2, decisions_between_saves=1).pick() is None


def test_delete_removes_every_saved_state(tmp_path):
    folder = tmp_path / "archive"
    archive = StateArchive(folder, capacity=2, decisions_between_saves=1)
    archive.visit((1,))
    save(archive, (1,))
    archive.delete()
    assert not folder.exists()
