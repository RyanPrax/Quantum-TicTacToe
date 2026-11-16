"""The permanent run store."""

import json
import subprocess

import pytest

from qttt.backends import IdealBackend
from qttt.board import O, X
from qttt.detector import detect_winner
from qttt.noise import calibrate
from qttt.results import RunStore, _git_state, provenance

SEED = 4903
TOP_ROW = (X, X, X, O, O, " ", " ", " ", " ")


@pytest.fixture
def store(tmp_path):
    return RunStore(tmp_path / "results")


def test_detection_is_saved_with_everything_needed(store):
    result = detect_winner(TOP_ROW, player=X, shots=128, backend=IdealBackend(seed=SEED), store=store)
    (record,) = store.load("detection")
    assert record["kind"] == "detection"
    assert record["board"] == "XXXOO    "
    assert record["player"] == X
    assert record["counts"] == list(result.run.counts)
    assert record["cutoffs"] == list(result.cutoffs)
    assert record["winner"] == X and record["lines"] == [0]
    assert record["backend"]["name"] == "ideal"
    assert record["metrics"]["line_qubits"] == [9, 10, 11]
    assert record["metrics"]["ancilla"] is False
    assert record["backend"]["options"]["ancilla"] == "auto"
    assert record["profile"]["source"] == "ideal"
    assert record["provenance"]["packages"]["qiskit"]


def test_board_with_three_true_lines_is_saved(store):
    detect_winner((X,) * 9, player=X, shots=16, backend=IdealBackend(seed=SEED), store=store)
    (record,) = store.load("detection")
    assert record["label"] == "multi_win" and len(record["true_lines"]) == 8


def test_circuit_round_trips(store):
    result = detect_winner(TOP_ROW, player=X, shots=16, backend=IdealBackend(seed=SEED), store=store)
    (record,) = store.load("detection")
    assert store.load_circuit(record) == result.run.transpiled


def test_store_only_appends(store):
    detect_winner(TOP_ROW, player=X, shots=16, backend=IdealBackend(seed=SEED), store=store)
    first = store.runs_file.read_text().splitlines()
    detect_winner(TOP_ROW, player=X, shots=16, backend=IdealBackend(seed=SEED), store=store)
    lines = store.runs_file.read_text().splitlines()
    assert len(lines) == 2 * len(first)
    assert lines[:len(first)] == first
    assert len({json.loads(line)["id"] for line in lines}) == len(lines)


def test_circuit_files_are_never_overwritten(store, monkeypatch):
    class SameId:
        hex = "0" * 32

    backend = IdealBackend(seed=SEED)
    result = detect_winner(TOP_ROW, player=X, shots=16, backend=backend)
    monkeypatch.setattr("qttt.results.uuid.uuid4", lambda: SameId)
    store.save_detection(result, backend)
    with pytest.raises(FileExistsError):
        store.save_detection(result, backend)


def test_store_path_can_be_a_string(tmp_path):
    detect_winner(TOP_ROW, player=X, shots=16, backend=IdealBackend(seed=SEED), store=str(tmp_path / "r"))
    assert len(RunStore(tmp_path / "r").load("detection")) == 1


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


def test_git_dirty_sees_changes_under_src(tmp_path):
    package = tmp_path / "src" / "pkg"
    package.mkdir(parents=True)
    (package / "code.py").write_text("x = 1\n")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(tmp_path)]
    for args in (["init", "-q"], ["add", "."], ["commit", "-qm", "init"]):
        subprocess.run(git + args, check=True)
    assert _git_state(package)["git_dirty"] is False
    (package / "code.py").write_text("x = 2\n")
    assert _git_state(package)["git_dirty"] is True


def test_records_never_hold_a_token(store, monkeypatch):
    monkeypatch.setenv("QISKIT_IBM_TOKEN", "secret-token-value")
    detect_winner(TOP_ROW, player=X, shots=16, backend=IdealBackend(seed=SEED), store=store)
    assert "secret-token-value" not in store.runs_file.read_text()


def test_a_job_saves_one_circuit_and_rebuilds_every_board(store):
    from qttt.detector import detect_winners

    boards = [TOP_ROW, (O, O, O, X, X, " ", X, " ", " "), (X, O, X, O, X, O, " ", " ", X)]
    players = [X, O, X]
    cases = list(zip(boards, players, strict=True))
    results = detect_winners(cases, shots=16, backend=IdealBackend(seed=SEED), store=store)
    assert len(list(store.circuits_dir.iterdir())) == 1  # the job's template only
    for record, result in zip(store.load("detection"), results, strict=True):
        assert "circuit_file" not in record and record["board_prep"]
        assert store.load_circuit(record) == result.run.transpiled


def test_board_gates_with_parameters_round_trip(tmp_path):
    # fake_rochester has no native X, so each board X becomes gates with angles.
    from qttt.backends import get_backend, with_board_prep

    rochester = get_backend("fake_rochester", seed=SEED)
    circuit = rochester.search_circuit("player", TOP_ROW, X)
    assert any(params for _, params, _ in circuit.metadata["board_prep"])
    assert with_board_prep(rochester.search_template("player"), circuit.metadata["board_prep"]) == circuit


def test_job_keeps_the_published_profile_and_profiles_round_trip(store):
    from qttt.noise import NoiseProfile

    result = detect_winner(TOP_ROW, player=X, shots=16, backend=IdealBackend(seed=SEED), store=store)
    (submitted,) = [r for r in store.load("job") if r["status"] == "submitted"]
    published = NoiseProfile.from_record(submitted["published_profile"])
    assert published.line_qubits == result.run.metrics.line_qubits  # measured on the job's own circuit
    assert NoiseProfile.from_record(store.load("detection")[0]["profile"]) == result.profile


def test_load_sees_records_appended_after_it_was_called(store):
    assert store.load() == []
    detect_winner((X, X, X, O, O, " ", " ", " ", " "), player=X, shots=64, backend=IdealBackend(seed=4903),
                  store=store)
    assert len(store.load("detection")) == 1
    other = RunStore(store.root)  # another store on the same files, as a second process would be
    detect_winner((X, X, X, O, O, " ", " ", " ", " "), player=X, shots=64, backend=IdealBackend(seed=4903),
                  store=other)
    assert len(store.load("detection")) == 2


def test_loaded_records_can_be_changed_without_changing_the_store(store):
    detect_winner((X, X, X, O, O, " ", " ", " ", " "), player=X, shots=64, backend=IdealBackend(seed=4903),
                  store=store)
    (record,) = store.load("detection")
    record["board"] = "changed"
    assert store.load("detection")[0]["board"] == "XXXOO    "


def test_provenance_reads_git_again_after_a_while(monkeypatch):
    from qttt import results

    calls = []
    monkeypatch.setattr(results, "_git_state", lambda: calls.append(1) or {"git_commit": "c", "git_dirty": False})
    monkeypatch.setattr(results, "_git_cache", None)
    clock = [100.0]
    monkeypatch.setattr(results.time, "monotonic", lambda: clock[0])
    provenance()
    provenance()
    assert len(calls) == 1  # one git call for a quick batch of records
    clock[0] += results.GIT_STATE_SECONDS + 1
    provenance()
    assert len(calls) == 2


def test_git_dirty_sees_changes_under_scripts(tmp_path):
    package = tmp_path / "src" / "pkg"
    package.mkdir(parents=True)
    (package / "code.py").write_text("x = 1\n")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run.py").write_text("y = 1\n")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(tmp_path)]
    for args in (["init", "-q"], ["add", "."], ["commit", "-qm", "init"]):
        subprocess.run(git + args, check=True)
    assert _git_state(package)["git_dirty"] is False
    (tmp_path / "scripts" / "run.py").write_text("y = 2\n")
    assert _git_state(package)["git_dirty"] is True


def test_dirty_runs_save_a_patch_that_rebuilds_the_code(tmp_path, monkeypatch):
    from qttt import results

    repo = tmp_path / "repo"
    package = repo / "src" / "pkg"
    package.mkdir(parents=True)
    (package / "code.py").write_text("x = 1\n")
    (repo / "notes.md").write_text("draft\n")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(repo)]
    for args in (["init", "-q"], ["add", "."], ["commit", "-qm", "init"]):
        subprocess.run(git + args, check=True)
    (repo / "notes.md").write_text("edited docs only\n")
    assert _git_state(package)["git_dirty"] is False  # docs do not count
    (package / "code.py").write_text("x = 2\n")
    (repo / "scripts").mkdir()
    (repo / "scripts" / "new.py").write_text("y = 1\n")  # not yet added to git
    monkeypatch.setattr(results, "_git_state", lambda: _git_state(package))
    monkeypatch.setattr(results, "_git_cache", None)
    store = RunStore(tmp_path / "results")
    first, second = (store._append({"kind": "test"}, None) for _ in range(2))
    records = {r["id"]: r["provenance"] for r in store.load()}
    assert records[first]["git_dirty"] and records[first]["patch"] == records[second]["patch"]  # saved once
    assert "diff" not in records[first]
    assert len(list(store.patches_dir.iterdir())) == 1
    patch = store.root / records[first]["patch"]
    subprocess.run(git + ["stash", "-qu"], check=True)  # back to the commit
    subprocess.run(git + ["apply", str(patch)], check=True)
    assert (package / "code.py").read_text() == "x = 2\n"
    assert (repo / "scripts" / "new.py").read_text() == "y = 1\n"


def test_clean_runs_save_no_patch(store, monkeypatch):
    from qttt import results

    monkeypatch.setattr(results, "_git_state", lambda: {"git_commit": "c", "git_dirty": False, "diff": ""})
    monkeypatch.setattr(results, "_git_cache", None)
    store._append({"kind": "test"}, None)
    assert "patch" not in store.load()[0]["provenance"] and not store.patches_dir.exists()
