"""Labeled data collection."""

import importlib.util
from pathlib import Path

import pytest

from qttt.backends import IdealBackend
from qttt.board import classical_winning_lines
from qttt.experiments import collect, sample_boards, summarize
from qttt.results import RunStore

SEED = 4903


def test_balanced_mix_and_true_labels():
    boards = sample_boards(10, seed=SEED)
    labels = [b.label for b in boards]
    assert sorted(labels.count(l) for l in ("no_win", "single_win", "double_win")) == [3, 3, 4]
    for b in boards:
        assert b.lines == classical_winning_lines(b.board, b.player)
        assert b.board.count(" ") <= 4  # turn 5 or later


def test_balanced_single_wins_cover_every_line_and_player():
    boards = sample_boards(seed=SEED, mix={"single_win": 16})
    assert sorted(b.stratum for b in boards) == [(line, p) for line in range(8) for p in ("O", "X")]
    first_eight = sample_boards(seed=SEED, mix={"single_win": 8})
    assert sorted(line for line, _ in (b.stratum for b in first_eight)) == list(range(8))
    assert {b.player for b in first_eight} == {"X", "O"}


def test_balanced_no_win_boards_cover_every_player_and_turn():
    boards = sample_boards(seed=SEED, mix={"no_win": 5})
    assert sorted(b.stratum for b in boards) == [("O", 6), ("O", 8), ("X", 5), ("X", 7), ("X", 9)]


def test_balanced_spreads_turns():
    turns = [b.turn for b in sample_boards(seed=SEED, mix={"single_win": 40})]
    assert set(turns) == {5, 6, 7, 8, 9}
    assert max(turns.count(t) for t in set(turns)) - min(turns.count(t) for t in set(turns)) <= 4


def test_same_seed_same_boards():
    assert sample_boards(9, seed=1) == sample_boards(9, seed=1)
    assert sample_boards(9, seed=1) != sample_boards(9, seed=2)


def test_natural_mix_follows_frequencies():
    boards = sample_boards(200, seed=SEED, mix="natural")
    counts = {l: sum(b.label == l for b in boards) for l in ("no_win", "single_win", "double_win")}
    assert sum(counts.values()) == 200
    assert counts == {"no_win": 157, "single_win": 42, "double_win": 1}


def test_explicit_mix_and_repeats_after_the_pool_runs_out():
    boards = sample_boards(seed=SEED, mix={"double_win": 30})
    assert len(boards) == 30
    assert len({b.board for b in boards}) == 22  # only 22 double wins exist


def test_bad_mix():
    with pytest.raises(ValueError):
        sample_boards(5, mix="skewed")
    with pytest.raises(ValueError):
        sample_boards(5, mix={"triple_win": 1})


def test_collect_saves_labeled_results(tmp_path):
    boards = sample_boards(6, seed=SEED)
    store = RunStore(tmp_path)
    results = collect(boards, IdealBackend(seed=SEED), shots=512, store=store, tag="test-batch")
    records = store.load("detection", tag="test-batch")
    assert len(records) == 6
    assert [r["label"] for r in records] == [b.label for b in boards]
    assert [r["true_lines"] for r in records] == [sorted(b.lines) for b in boards]
    assert [r["turn"] for r in records] == [b.turn for b in boards]
    s = summarize(boards, results)
    assert s.wins_found["single_win"] == s.boards["single_win"]
    assert s.all_lines_found["double_win"] == s.boards["double_win"]
    assert "single_win" in s.text()


def test_collect_splits_into_jobs(tmp_path):
    class Counting(IdealBackend):
        jobs = []

        def run(self, circuits, shots, on_submit=None):
            self.jobs.append(len(circuits))
            return super().run(circuits, shots, on_submit)

    backend = Counting(seed=SEED)
    collect(sample_boards(5, seed=SEED), backend, shots=64, store=RunStore(tmp_path), boards_per_job=2,
            profile="calibrate")
    assert backend.jobs == [9 + 2, 9 + 2, 9 + 1]  # each job calibrates its own boards


def test_default_tag_is_a_timestamp(tmp_path):
    store = RunStore(tmp_path)
    collect(sample_boards(1, seed=SEED), IdealBackend(seed=SEED), shots=64, store=store)
    assert store.load()[0]["tag"].startswith("collect-")


def test_collect_script(tmp_path, capsys):
    path = Path(__file__).resolve().parent.parent / "scripts" / "collect.py"
    spec = importlib.util.spec_from_file_location("collect_script", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert script.main(["--n", "3", "--seed", "1", "--shots", "128", "--store", str(tmp_path), "--tag", "cli"]) == 0
    out = capsys.readouterr().out
    assert "3 boards" in out and "Saved 3 results" in out
    assert len(RunStore(tmp_path).load("detection", tag="cli")) == 3


def test_collect_script_compile_flags(tmp_path, capsys):
    path = Path(__file__).resolve().parent.parent / "scripts" / "collect.py"
    spec = importlib.util.spec_from_file_location("collect_script", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert script.main(["--n", "1", "--seed", "1", "--shots", "64", "--store", str(tmp_path), "--ancilla", "on"]) == 0
    (record,) = RunStore(tmp_path).load("detection")
    assert record["metrics"]["ancilla"] is True
    assert record["backend"]["options"]["ancilla"] is True


def test_mix_mapping_must_match_n():
    assert len(sample_boards(3, seed=SEED, mix={"single_win": 3})) == 3
    with pytest.raises(ValueError, match="adds up to 3"):
        sample_boards(5, seed=SEED, mix={"single_win": 3})


def test_collect_nothing(tmp_path):
    assert collect([], IdealBackend(seed=SEED), store=RunStore(tmp_path)) == []
    assert not (tmp_path / "runs.jsonl").exists()
    with pytest.raises(ValueError, match="boards_per_job"):
        collect(sample_boards(1, seed=SEED), IdealBackend(seed=SEED), store=RunStore(tmp_path), boards_per_job=0)
