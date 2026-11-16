"""Labeled data collection."""

import argparse
import importlib.util
from pathlib import Path

import pytest

from qttt.backends import IdealBackend
from qttt.board import classical_winning_lines
from qttt.experiments import collect, sample_boards, summarize
from qttt.noise import CALIBRATION_CIRCUITS
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
    assert backend.jobs == [CALIBRATION_CIRCUITS + 2, CALIBRATION_CIRCUITS + 2, CALIBRATION_CIRCUITS + 1]


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


def test_collect_script_dry_run_without_a_mock_device(capsys):
    # IBM dry-runs on "mock_<device>"; an account that cannot see it gets a message, not a traceback.
    from qiskit.providers.exceptions import QiskitBackendNotFoundError

    class NoMockDevice:
        name, simulated = "ibm_marrakesh", False

        def dry_run(self, circuits, shots):
            raise QiskitBackendNotFoundError("Unable to use `dry_run` mode: backend 'mock_marrakesh' was not found")

    path = Path(__file__).resolve().parent.parent / "scripts" / "collect.py"
    spec = importlib.util.spec_from_file_location("collect_script", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert script._dry_run(NoMockDevice(), [], argparse.Namespace(variant="player"), 2048, calibrate=False) == 1
    out = capsys.readouterr().out
    assert "could not dry-run on ibm_marrakesh" in out and "Nothing was sent." in out


def test_mix_mapping_must_match_n():
    assert len(sample_boards(3, seed=SEED, mix={"single_win": 3})) == 3
    with pytest.raises(ValueError, match="adds up to 3"):
        sample_boards(5, seed=SEED, mix={"single_win": 3})


def test_collect_nothing(tmp_path):
    assert collect([], IdealBackend(seed=SEED), store=RunStore(tmp_path)) == []
    assert not (tmp_path / "runs.jsonl").exists()
    with pytest.raises(ValueError, match="boards_per_job"):
        collect(sample_boards(1, seed=SEED), IdealBackend(seed=SEED), store=RunStore(tmp_path), boards_per_job=0)


def test_choose_shots_calibrates_once_and_saves_it(tmp_path):
    from qttt.experiments import choose_shots

    store = RunStore(tmp_path)
    shots, profile = choose_shots(IdealBackend(seed=SEED), store=store, tag="auto")
    assert 64 <= shots <= 1024  # a perfect simulator needs few shots
    assert shots == profile.shots_needed(0.01, profile_budget(), 4096, in_job=True)
    (calibration,) = store.load("calibration", tag="auto")
    assert len(calibration["known"]) == len(calibration["profile"]["readout"]) + 4  # 8 single, 2 no-win, 2 double


def profile_budget():
    from qttt.detector import DEFAULT_BUDGET
    return DEFAULT_BUDGET


def test_choose_shots_stops_at_the_cap(tmp_path, monkeypatch):
    from qttt.experiments import choose_shots
    from qttt.noise import NoiseProfile

    monkeypatch.setattr(NoiseProfile, "shots_needed", lambda self, *a, **k: None)  # some line undetectable
    shots, _ = choose_shots(IdealBackend(seed=SEED), max_shots=2048, store=RunStore(tmp_path))
    assert shots == 2048


def test_collect_script_chooses_shots(tmp_path, capsys):
    path = Path(__file__).resolve().parent.parent / "scripts" / "collect.py"
    spec = importlib.util.spec_from_file_location("collect_script", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert script.main(["--n", "3", "--seed", "1", "--shots", "auto", "--store", str(tmp_path), "--tag", "a"]) == 0
    out = capsys.readouterr().out
    assert "Choosing shots" in out and "shots per board" in out
    store = RunStore(tmp_path)
    assert len(store.load("calibration", tag="a")) == 1 and len(store.load("detection", tag="a")) == 3


def _two_saved_calibrations(store):
    from qttt.experiments import choose_shots

    for tag in ("cal-1", "cal-2"):
        choose_shots(IdealBackend(seed=SEED), store=store, tag=tag)


def test_saved_calibrations_replace_the_batch_calibration(tmp_path):
    from qttt.experiments import save_pooled_calibration, saved_profile

    store = RunStore(tmp_path)
    _two_saved_calibrations(store)

    class Counting(IdealBackend):
        jobs = []

        def run(self, circuits, shots, on_submit=None):
            self.jobs.append(len(circuits))
            return super().run(circuits, shots, on_submit)

    backend = Counting(seed=SEED)
    profile, records = saved_profile(store, ["cal-1", "cal-2"], backend)
    assert len(records) == 2 and profile.no_win_shares is not None
    profile = save_pooled_calibration(store, profile, records, backend, tag="batch")
    collect(sample_boards(3, seed=SEED), backend, shots=64, profile=profile, store=store, tag="batch")
    assert backend.jobs == [3]  # the boards only: no calibration circuits
    (pooled,) = store.load("calibration", tag="batch")
    assert pooled["pooled_from"] == [r["id"] for r in records] and pooled["job_id"] is None
    assert {d["calibration_id"] for d in store.load("detection", tag="batch")} == {pooled["id"]}


def test_saved_calibrations_that_do_not_fit_are_refused(tmp_path):
    from qttt.experiments import saved_profile
    from qttt.noise import NoiseProfile

    store = RunStore(tmp_path)
    _two_saved_calibrations(store)

    class Other(IdealBackend):
        name = "other"

    class Suppressed(IdealBackend):
        def options(self):
            return {**super().options(), "sampler_options": {"twirling": {"enable_gates": True}}}

    with pytest.raises(ValueError, match="ran on ideal"):
        saved_profile(store, ["cal-1"], Other(seed=SEED))
    with pytest.raises(ValueError, match="Sampler options"):
        saved_profile(store, ["cal-1"], Suppressed(seed=SEED))
    with pytest.raises(ValueError, match="no calibration saved with tag missing"):
        saved_profile(store, ["missing"], IdealBackend(seed=SEED))
    (record,) = store.load("calibration", tag="cal-1")
    moved = NoiseProfile.from_record({**record["profile"], "line_qubits": [0, 1, 2]})
    from qttt.backends import JobOutput
    counts = tuple(map(tuple, record["readout_counts"])) + tuple(tuple(k["counts"]) for k in record["known"])
    store.save_calibration(moved, JobOutput(counts, None), IdealBackend(seed=SEED),
                           [(k["board"], k["player"]) for k in record["known"]], tag="moved")
    with pytest.raises(ValueError, match="reads out"):
        saved_profile(store, ["moved"], IdealBackend(seed=SEED))


def test_collect_script_uses_saved_calibrations(tmp_path, capsys):
    path = Path(__file__).resolve().parent.parent / "scripts" / "collect.py"
    spec = importlib.util.spec_from_file_location("collect_script", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    store = RunStore(tmp_path)
    _two_saved_calibrations(store)
    assert script.main(["--n", "3", "--seed", "1", "--shots", "auto", "--store", str(tmp_path), "--tag", "b",
                        "--calibration", "cal-1", "cal-2"]) == 0
    out = capsys.readouterr().out
    assert "sends no calibration circuits" in out and "Shots from that calibration" in out
    assert len(store.load("detection", tag="b")) == 3


def test_balanced_mix_uses_every_double_win_twice_then_splits_evenly():
    from collections import Counter

    boards = sample_boards(1000, seed=SEED)
    labels = Counter(b.label for b in boards)
    assert labels == {"double_win": 44, "single_win": 478, "no_win": 478}
    doubles = Counter(b.board for b in boards if b.label == "double_win")
    assert len(doubles) == 22 and set(doubles.values()) == {2}
    assert Counter(b.label for b in sample_boards(9, seed=SEED)) == {"no_win": 3, "single_win": 3, "double_win": 3}


def _save_variant_of(store, record, tag, **profile_fields):
    from qttt.backends import JobOutput
    from qttt.noise import NoiseProfile

    profile = NoiseProfile.from_record({**record["profile"], **profile_fields})
    counts = tuple(map(tuple, record["readout_counts"])) + tuple(tuple(k["counts"]) for k in record["known"])
    store.save_calibration(profile, JobOutput(counts, None), IdealBackend(seed=SEED),
                           [(k["board"], k["player"]) for k in record["known"]], tag=tag)


def test_saved_calibrations_of_another_compiled_circuit_are_refused(tmp_path):
    from qttt.experiments import saved_profile

    store = RunStore(tmp_path)
    _two_saved_calibrations(store)
    (record,) = store.load("calibration", tag="cal-1")
    assert record["profile"]["template"]["gates_sha256"]
    _save_variant_of(store, record, "other-circuit", template={**record["profile"]["template"], "gates_sha256": "0"})
    with pytest.raises(ValueError, match="compiled circuit differs"):
        saved_profile(store, ["other-circuit"], IdealBackend(seed=SEED))
    # A different transpile seed that found the same circuit still fits.
    _save_variant_of(store, record, "other-seed", template={**record["profile"]["template"], "transpile_seed": 99})
    saved_profile(store, ["other-seed"], IdealBackend(seed=SEED))


def test_old_calibrations_without_a_template_or_line_qubits(tmp_path):
    from qttt.experiments import saved_profile

    store = RunStore(tmp_path)
    _two_saved_calibrations(store)
    (record,) = store.load("calibration", tag="cal-1")
    _save_variant_of(store, record, "no-template", template=None)
    with pytest.warns(UserWarning, match="without their compiled circuit"):
        saved_profile(store, ["no-template"], IdealBackend(seed=SEED))
    _save_variant_of(store, record, "no-qubits", line_qubits=None)
    with pytest.raises(ValueError, match="does not say which line qubits"):
        saved_profile(store, ["no-qubits"], IdealBackend(seed=SEED))


def test_auto_shots_still_calibrate_in_every_batch_job(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parent.parent / "scripts" / "collect.py"
    spec = importlib.util.spec_from_file_location("collect_script", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    passed = {}
    real_collect = script.collect

    def spy(*args, **kwargs):
        passed.update(kwargs)
        return real_collect(*args, **kwargs)
    monkeypatch.setattr(script, "collect", spy)
    assert script.main(["--n", "2", "--seed", "1", "--shots", "auto", "--store", str(tmp_path), "--tag", "auto"]) == 0
    # The calibration that chose the shots is not reused: each batch job calibrates at the chosen shots (on a
    # noisy backend), as the message about calibration circuits per job says.
    assert passed["profile"] == "auto"
    (calibration,) = RunStore(tmp_path).load("calibration", tag="auto")  # the one that chose the shots


@pytest.mark.parametrize("argv", [["--n", "0"], ["--shots", "0"], ["--boards-per-job", "0"], ["--n", "x"]])
def test_collect_script_refuses_bad_numbers(argv, tmp_path):
    path = Path(__file__).resolve().parent.parent / "scripts" / "collect.py"
    spec = importlib.util.spec_from_file_location("collect_script", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    with pytest.raises(SystemExit) as info:
        script.main([*argv, "--store", str(tmp_path)])
    assert info.value.code == 2  # an argparse error, not a traceback
