"""The permanent run store."""

import json

import pytest

from qttt.backends import IdealBackend
from qttt.board import O, X
from qttt.detector import detect_winner
from qttt.noise import calibrate
from qttt.results import RunStore, provenance

SEED = 4903
TOP_ROW = (X, X, X, O, O, " ", " ", " ", " ")


@pytest.fixture
def store(tmp_path):
    return RunStore(tmp_path / "results")


def test_detection_is_saved_with_everything_needed(store):
    result = detect_winner(TOP_ROW, player=X, shots=128, backend=IdealBackend(seed=SEED), store=store)
    (record,) = store.load()
    assert record["kind"] == "detection"
    assert record["board"] == "XXXOO    "
    assert record["player"] == X
    assert record["counts"] == list(result.run.counts)
    assert record["cutoffs"] == list(result.cutoffs)
    assert record["winner"] == X and record["lines"] == [0]
    assert record["backend"]["name"] == "ideal"
    assert record["metrics"]["line_qubits"] == [9, 10, 11]
    assert record["profile"]["source"] == "ideal"
    assert record["provenance"]["packages"]["qiskit"]


def test_board_with_three_true_lines_is_saved(store):
    detect_winner((X,) * 9, player=X, shots=16, backend=IdealBackend(seed=SEED), store=store)
    (record,) = store.load()
    assert record["label"] == "multi_win" and len(record["true_lines"]) == 8


def test_circuit_round_trips(store):
    result = detect_winner(TOP_ROW, player=X, shots=16, backend=IdealBackend(seed=SEED), store=store)
    (record,) = store.load()
    assert store.load_circuit(record) == result.run.transpiled


def test_store_only_appends(store):
    detect_winner(TOP_ROW, player=X, shots=16, backend=IdealBackend(seed=SEED), store=store)
    first_line = store.runs_file.read_text().splitlines()[0]
    detect_winner(TOP_ROW, player=X, shots=16, backend=IdealBackend(seed=SEED), store=store)
    lines = store.runs_file.read_text().splitlines()
    assert len(lines) == 2
    assert lines[0] == first_line
    assert len({json.loads(line)["id"] for line in lines}) == 2


def test_circuit_files_are_never_overwritten(store, monkeypatch):
    class SameId:
        hex = "0" * 32

    monkeypatch.setattr("qttt.results.uuid.uuid4", lambda: SameId)
    detect_winner(TOP_ROW, player=X, shots=16, backend=IdealBackend(seed=SEED), store=store)
    with pytest.raises(FileExistsError):
        detect_winner(TOP_ROW, player=X, shots=16, backend=IdealBackend(seed=SEED), store=store)


def test_store_path_can_be_a_string(tmp_path):
    detect_winner(TOP_ROW, player=X, shots=16, backend=IdealBackend(seed=SEED), store=str(tmp_path / "r"))
    assert len(RunStore(tmp_path / "r").load()) == 1


def test_simulator_runs_are_not_saved_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    detect_winner(TOP_ROW, player=X, shots=16, backend=IdealBackend(seed=SEED))
    assert not (tmp_path / "results").exists()


def test_calibration_is_saved(store):
    backend = IdealBackend(seed=SEED)
    known = ((TOP_ROW, X),)
    profile, out = calibrate(backend, shots=64, known=known)
    store.save_calibration(profile, out, backend, known)
    (record,) = store.load("calibration")
    assert len(record["readout_counts"]) == 8
    assert record["known"][0]["board"] == "XXXOO    "
    assert record["profile"]["epsilon"] == profile.epsilon
    assert store.load("detection") == []


def test_load_on_empty_store(store):
    assert store.load() == []


def test_provenance_records_the_code_version():
    info = provenance()
    assert "git_commit" in info and "git_dirty" in info
    assert info["packages"]["qiskit"]


def test_records_never_hold_a_token(store, monkeypatch):
    monkeypatch.setenv("QISKIT_IBM_TOKEN", "secret-token-value")
    detect_winner(TOP_ROW, player=X, shots=16, backend=IdealBackend(seed=SEED), store=store)
    assert "secret-token-value" not in store.runs_file.read_text()
