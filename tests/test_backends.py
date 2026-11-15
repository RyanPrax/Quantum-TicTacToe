"""Backends: template transpiling, readout circuits, running and the IBM account path.

Fake-backend runs simulate the device's noise and take a few seconds each;
they are marked slow (skip with -m "not slow").
"""

import random

import pytest
from qiskit_aer.primitives import SamplerV2 as AerSampler

from qttt.backends import IBMBackend, IdealBackend, get_backend, line_qubits
from qttt.board import O, X, boards_by_win_count, classical_winning_lines
from qttt.detector import detect_winner
from qttt.grover import build_search_circuit
from qttt.noise import NoiseProfile, ideal_line_probabilities
from qttt.results import RunStore

SEED = 4903
TOP_ROW = (X, X, X, O, O, " ", " ", " ", " ")

_GROUPS = boards_by_win_count()
_rng = random.Random(SEED)
BOARDS = _rng.sample(_GROUPS[0], 3) + _rng.sample(_GROUPS[1], 3) + _GROUPS[2][:2]


def _last_player(board):
    return X if board.count(X) > board.count(O) else O


@pytest.fixture(scope="module")
def berlin():
    return get_backend("fake_berlin", seed=SEED)


def test_get_backend_ideal():
    backend = get_backend("ideal", seed=1)
    assert isinstance(backend, IdealBackend)
    assert backend.simulated and backend.seed == 1


def test_get_backend_fake_is_simulated(berlin):
    assert isinstance(berlin, IBMBackend)
    assert berlin.simulated
    assert berlin.calibration_date() is not None


def test_line_qubits_of_logical_circuit():
    # Board (9) then line (3): the line register is qubits 9-11.
    assert line_qubits(build_search_circuit("player", TOP_ROW, X)) == (9, 10, 11)


def test_every_board_gets_the_same_routed_circuit(berlin):
    metrics = {berlin.metrics(berlin.search_circuit("player", b, _last_player(b))) for b in BOARDS}
    assert len({(m.two_qubit_gates, m.line_qubits) for m in metrics}) == 1


@pytest.mark.parametrize("variant", ["player", "full"])
def test_transpiled_circuits_still_find_the_winner(berlin, variant):
    # Run the hardware-transpiled circuits without noise: the answers must not change.
    boards = BOARDS if variant == "player" else BOARDS[3:5]
    circuits = [berlin.search_circuit(variant, b, _last_player(b)) for b in boards]
    _assert_finds_winners(boards, circuits)


def test_transpiled_circuits_with_ancilla_still_find_the_winner():
    backend = get_backend("fake_berlin", seed=SEED, ancilla=True)
    assert backend.logical_search_circuit("player", TOP_ROW, X).num_qubits == 13
    assert backend.logical_search_circuit("full", TOP_ROW).num_qubits == 22  # full ignores the setting
    assert backend.options()["ancilla"] is True
    _assert_finds_winners(BOARDS, [backend.search_circuit("player", b, _last_player(b)) for b in BOARDS])


def test_auto_ancilla_keeps_the_better_template(berlin):
    assert berlin.ancilla == "auto"
    forced = [get_backend("fake_berlin", seed=SEED, ancilla=a) for a in (False, True)]
    scores = [b.score(b.search_template("player")) for b in forced]
    chosen = berlin.search_template("player")
    assert berlin.score(chosen) == max(scores)
    assert berlin.uses_ancilla("player") == (scores[1] > scores[0]) == berlin.metrics(chosen).ancilla
    assert not berlin.uses_ancilla("full")


def test_auto_ancilla_on_ideal_backend_uses_no_ancilla():
    backend = IdealBackend(seed=SEED)
    assert not backend.uses_ancilla("player")
    assert backend.logical_search_circuit("player", TOP_ROW, X).num_qubits == 12


def test_unknown_ancilla_setting_is_refused():
    with pytest.raises(ValueError, match="ancilla"):
        IdealBackend(ancilla="yes")


def _assert_finds_winners(boards, circuits):
    results = AerSampler(seed=SEED).run(circuits, shots=4000).result()
    for board, result in zip(boards, results, strict=True):
        counts = result.data.line_cl.get_counts()
        measured = [counts.get(format(i, "03b"), 0) / 4000 for i in range(8)]
        winners = classical_winning_lines(board, _last_player(board))
        assert measured == pytest.approx(ideal_line_probabilities(winners), abs=0.04)


def test_more_transpile_seeds_never_hurt():
    one = get_backend("fake_berlin", seed=SEED, transpile_seeds=1)
    four = get_backend("fake_berlin", seed=SEED, transpile_seeds=4)
    esp_one = one.metrics(one.search_template("player")).log10_esp
    esp_four = four.metrics(four.search_template("player")).log10_esp
    assert esp_four >= esp_one


def test_readout_circuits_use_the_given_qubits(berlin):
    qubits = berlin.metrics(berlin.search_template("player")).line_qubits
    for circuit in berlin.readout_circuits(qubits):
        assert line_qubits(circuit) == qubits


def test_ideal_readout_circuits_read_back_each_index():
    backend = IdealBackend(seed=SEED)
    out = backend.run(backend.readout_circuits((9, 10, 11)), shots=100)
    for i, counts in enumerate(out.counts):
        assert counts[i] == 100


def test_published_readout_errors(berlin):
    errors = berlin.readout_errors((0, 1, 2))
    assert len(errors) == 3
    assert all(0 <= p < 0.5 for pair in errors for p in pair)


@pytest.mark.slow
def test_detect_on_fake_backend_calibrates_by_default(berlin):
    result = detect_winner(TOP_ROW, player=X, shots=512, backend=berlin)
    assert result.profile.source == "calibration:fake_berlin"
    assert 0 < result.profile.epsilon < 1
    assert result.lines <= {0}


@pytest.mark.slow
def test_detect_on_fake_backend_with_profile(berlin):
    profile = NoiseProfile.from_backend(berlin)
    result = detect_winner(TOP_ROW, player=X, shots=256, backend=berlin, profile=profile)
    assert result.lines <= {0}
    assert result.run.backend == "fake_berlin"
    assert result.cutoffs == profile.cutoffs(256, 0.1)


def test_profile_from_other_qubits_is_refused_before_running(berlin, monkeypatch):
    monkeypatch.setattr("qttt.backends.Backend.run", lambda *a, **k: pytest.fail("ran"))
    profile = NoiseProfile(0.5, NoiseProfile.from_ideal().readout, "test", line_qubits=(0, 1, 2))
    with pytest.raises(ValueError, match="line qubits"):
        detect_winner(TOP_ROW, player=X, shots=16, backend=berlin, profile=profile)


@pytest.mark.slow
def test_hardware_path_runs_locally_and_always_saves(tmp_path, monkeypatch):
    # IBM's runtime sampler runs a fake backend locally, so this exercises the
    # real-hardware code path (job id, automatic saving) without a network.
    from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

    monkeypatch.chdir(tmp_path)
    device = IBMBackend(FakeProviderForBackendV2().backend("fake_berlin"), seed=SEED, simulate=False)
    result = detect_winner(TOP_ROW, player=X, shots=32, backend=device)
    assert result.run.job_id
    store = RunStore(tmp_path / "results")
    (calibration,) = store.load("calibration")  # calibrated in the same job by default
    (detection,) = store.load("detection")
    assert detection["job_id"] == result.run.job_id == calibration["job_id"]
    assert detection["calibration_id"] == calibration["id"]
    submitted, done = store.job(result.run.job_id)  # recorded on submit, counts recorded on return
    assert detection["job_record"] == submitted["id"] and done["job_id"] == result.run.job_id
    assert detection["backend"]["simulated"] is False
    assert result.profile.source == "calibration:fake_berlin"


def test_read_dotenv_reads_only_account_keys_from_nearest_file(tmp_path, monkeypatch):
    from qttt.backends import read_dotenv

    (tmp_path / ".env").write_text(
        "# comment\n\nexport QISKIT_IBM_TOKEN='quoted value'\nQISKIT_IBM_INSTANCE=crn:with=equals\n"
        "OTHER_SECRET=never-read\nnot a line\n"
    )
    child = tmp_path / "notebooks"
    child.mkdir()
    monkeypatch.delenv("OTHER_SECRET", raising=False)
    assert read_dotenv(child) == {"QISKIT_IBM_TOKEN": "quoted value", "QISKIT_IBM_INSTANCE": "crn:with=equals"}
    import os
    assert "OTHER_SECRET" not in os.environ  # nothing is written to the environment


def test_read_dotenv_without_file(tmp_path, monkeypatch):
    from qttt import backends

    monkeypatch.setattr(backends.Path, "is_file", lambda self: False)  # no .env anywhere up the tree
    assert backends.read_dotenv(tmp_path) == {}


def _fake_service(monkeypatch):
    import qiskit_ibm_runtime

    seen = {}

    class FakeService:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(qiskit_ibm_runtime, "QiskitRuntimeService", FakeService)
    return seen


def test_runtime_service_reads_token_from_dotenv(tmp_path, monkeypatch):
    from qttt import backends

    seen = _fake_service(monkeypatch)
    monkeypatch.delenv("QISKIT_IBM_TOKEN", raising=False)
    monkeypatch.delenv("QISKIT_IBM_INSTANCE", raising=False)
    (tmp_path / ".env").write_text("QISKIT_IBM_TOKEN=token-from-dotenv\nQISKIT_IBM_INSTANCE=\n")
    monkeypatch.chdir(tmp_path)
    backends._runtime_service()
    assert seen == {"channel": "ibm_quantum_platform", "token": "token-from-dotenv", "instance": None}
    import os
    assert "QISKIT_IBM_TOKEN" not in os.environ  # the token never leaks into later tests


def test_runtime_service_prefers_the_environment(tmp_path, monkeypatch):
    from qttt import backends

    seen = _fake_service(monkeypatch)
    (tmp_path / ".env").write_text("QISKIT_IBM_TOKEN=token-from-dotenv\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QISKIT_IBM_TOKEN", "token-from-environment")
    backends._runtime_service()
    assert seen["token"] == "token-from-environment"


def test_runtime_service_reads_token_from_environment(monkeypatch):
    from qttt import backends

    seen = _fake_service(monkeypatch)
    monkeypatch.setattr(backends, "read_dotenv", lambda: {})  # ignore any real .env
    monkeypatch.setenv("QISKIT_IBM_TOKEN", "not-a-real-token")
    monkeypatch.setenv("QISKIT_IBM_INSTANCE", "my-instance")
    backends._runtime_service()
    assert seen == {"channel": "ibm_quantum_platform", "token": "not-a-real-token", "instance": "my-instance"}

    seen.clear()
    monkeypatch.delenv("QISKIT_IBM_TOKEN")
    backends._runtime_service()
    assert "token" not in seen  # falls back to the saved account


def test_board_gates_are_translated_for_old_devices():
    # fake_rochester has no native X gate (u1/u2/u3/cx); the board's X gates must be rewritten.
    rochester = get_backend("fake_rochester", seed=SEED)
    circuit = rochester.search_circuit("player", TOP_ROW, X)
    assert all(inst.operation.name in rochester.backend.target.operation_names for inst in circuit.data
               if inst.operation.name not in ("barrier", "measure"))
    counts = AerSampler(seed=SEED).run([circuit], shots=4000).result()[0].data.line_cl.get_counts()
    assert counts["000"] / 4000 == pytest.approx(25 / 32, abs=0.04)


@pytest.mark.parametrize("schedule", [True, False])
def test_simulated_runs_pad_idle_time_with_delays(berlin, monkeypatch, schedule):
    seen = []

    class Recorder:
        def run(self, circuits, shots):
            seen.extend(circuits)
            raise StopIteration

    monkeypatch.setattr("qttt.backends.AerSampler.from_backend", lambda *a, **k: Recorder())
    backend = IBMBackend(berlin.backend, seed=SEED, schedule=schedule)
    with pytest.raises(StopIteration):
        backend.run([backend.search_circuit("player", TOP_ROW, X)], shots=8)
    assert ("delay" in seen[0].count_ops()) == schedule
    assert backend.options()["schedule"] is schedule
