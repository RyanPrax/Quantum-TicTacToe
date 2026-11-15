"""Tests for the winner detector.

Cutoff and flag-and-check tests are pure Python. End-to-end tests run the
search circuit on the ideal simulator with a fixed seed, so their counts are
the same on every run; the legacy 22-qubit variant takes about a second per
board, so its tests are marked slow.
"""

import dataclasses
import math
import random

import pytest
from qiskit import QuantumCircuit

from qttt.backends import IdealBackend
from qttt.board import EMPTY, WIN_LINES, O, X, boards_by_win_count, classical_winning_lines
from qttt.detector import (
    DEFAULT_BUDGET,
    DEFAULT_SHOTS,
    NUM_LINES,
    DetectionResult,
    SearchRun,
    detect_winner,
    detect_winners,
    flag_and_check,
    resume_job,
    run_search,
)
from qttt.noise import CALIBRATION_CIRCUITS
from qttt.results import RunStore
from qttt.stats import binomial_cutoff, binomial_log_pmf, binomial_tail

SAMPLE_SIZE = 10
SEED = 4903

_ = EMPTY
TOP_ROW = (X, X, X,
           O, O, _,
           _, _, _)

_GROUPS = boards_by_win_count()
_rng = random.Random(SEED)
NO_WIN_SAMPLE = _rng.sample(_GROUPS[0], SAMPLE_SIZE)
SINGLE_WIN_SAMPLE = _rng.sample(_GROUPS[1], SAMPLE_SIZE)
DOUBLE_WINS = _GROUPS[2]


def _board_id(board):
    return "".join(board).replace(" ", ".")


def _winners(board):
    return classical_winning_lines(board, X) | classical_winning_lines(board, O)


def _reference_cutoff(shots, budget, q0):
    """Straight from the definition: count up from 0 until within budget."""
    k = 0
    while NUM_LINES * binomial_tail(k, shots, q0) > budget:
        k += 1
    return k


# --- Binomial helpers ---

def test_log_pmf_matches_direct_formula():
    for j in range(5):
        direct = math.comb(4, j) * (1 / 8) ** j * (7 / 8) ** (4 - j)
        assert math.exp(binomial_log_pmf(j, 4, 1 / 8)) == pytest.approx(direct)


@pytest.mark.parametrize("n", [1, 10, 1024])
def test_pmf_sums_to_one(n):
    total = sum(math.exp(binomial_log_pmf(j, n, 1 / 8)) for j in range(n + 1))
    assert total == pytest.approx(1.0)


def test_tail_small_example():
    # 4 shots: P(K >= 2) = 0.072 + 0.0068 + 0.00024
    assert binomial_tail(2, 4, 1 / 8) == pytest.approx(0.07886, abs=1e-5)


def test_tail_edges():
    assert binomial_tail(0, 10, 0.3) == 1.0
    assert binomial_tail(-1, 10, 0.3) == 1.0
    assert binomial_tail(11, 10, 0.3) == 0.0


# --- binomial_cutoff ---

@pytest.mark.parametrize(
    ("shots", "budget", "expected"),
    [
        (4, 0.1, 3),
        (100, 0.1, 21),
        (1024, 0.1, 153),
        (4096, 0.1, 561),
        (1024, 0.01, 162),
        (1024, 1.0, 141),
    ],
)
def test_cutoff_table_values(shots, budget, expected):
    assert binomial_cutoff(shots, budget) == expected


@pytest.mark.parametrize("shots", [4, 100, 1024, 4096])
def test_cutoff_is_smallest_within_budget(shots):
    k = binomial_cutoff(shots)
    assert NUM_LINES * binomial_tail(k, shots, 1 / 8) <= DEFAULT_BUDGET
    assert NUM_LINES * binomial_tail(k - 1, shots, 1 / 8) > DEFAULT_BUDGET


@pytest.mark.parametrize("q0", [0.05, 1 / 8, 0.2])
@pytest.mark.parametrize("budget", [0.01, 0.1, 1.0])
def test_cutoff_matches_reference(budget, q0):
    for shots in range(1, 151):
        assert binomial_cutoff(shots, budget, q0) == _reference_cutoff(shots, budget, q0)


def test_defaults():
    assert binomial_cutoff() == binomial_cutoff(1024, DEFAULT_BUDGET, 1 / 8)


def test_more_shots_lowers_cutoff_fraction():
    fractions = [binomial_cutoff(n) / n for n in (100, 1024, 4096)]
    assert fractions == sorted(fractions, reverse=True)
    assert all(f > 1 / 8 for f in fractions)


def test_bigger_budget_never_raises_cutoff():
    cutoffs = [binomial_cutoff(1024, b) for b in (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 8.0)]
    assert cutoffs == sorted(cutoffs, reverse=True)


def test_higher_q0_raises_cutoff():
    assert binomial_cutoff(1024, q0=0.10) < binomial_cutoff(1024, q0=1 / 8) < binomial_cutoff(1024, q0=0.15)


def test_too_few_shots_never_flags():
    # 1 shot: P(K >= 1) = 1/8, so 8 * 1/8 = 1 > 0.1. Only k = 2 (impossible) fits.
    assert binomial_cutoff(1) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shots": 0},
        {"shots": -5},
        {"budget": 0},
        {"budget": -0.1},
        {"budget": NUM_LINES + 0.1},
        {"q0": 0},
        {"q0": 1},
        {"q0": 1.5},
    ],
)
def test_cutoff_rejects_bad_input(kwargs):
    with pytest.raises(ValueError):
        binomial_cutoff(**kwargs)


# --- SearchRun and DetectionResult ---

def _run(counts=(900, 20, 20, 20, 16, 16, 16, 16)):
    return SearchRun(sum(counts), tuple(counts), QuantumCircuit(1))


def test_search_run_probabilities():
    run = _run()
    assert run.probabilities[0] == pytest.approx(900 / 1024)
    assert sum(run.probabilities) == pytest.approx(1.0)
    assert all(isinstance(p, float) for p in run.probabilities)


def test_detection_result_properties():
    run = _run()
    result = DetectionResult(
        board=TOP_ROW, variant="player", player=X, winner=X, lines=frozenset({0}),
        candidates=(0, 7), rejected=(7,), cutoffs=(153,) * 8, budget=0.1, run=run,
    )
    assert result.probabilities == run.probabilities
    assert result.checks == 2


def test_results_are_frozen():
    run = _run()
    with pytest.raises(dataclasses.FrozenInstanceError):
        run.shots = 1
    result = DetectionResult(TOP_ROW, "player", X, None, frozenset(), (), (), (153,) * 8, 0.1, run)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.winner = X


# --- flag_and_check (made-up counts, no circuit) ---

DOUBLE_WIN = (X, X, X,
              X, O, O,
              X, O, O)


@pytest.mark.parametrize(
    ("board", "counts", "expected"),
    [
        # One real winner.
        (TOP_ROW, (800, 30, 30, 30, 30, 30, 30, 44), ((0,), (), frozenset({0}), X)),
        # Real winner plus a false flag on line 7.
        (TOP_ROW, (600, 30, 30, 30, 30, 30, 30, 244), ((0, 7), (7,), frozenset({0}), X)),
        # Nothing reaches the cutoff.
        (TOP_ROW, (100, 130, 130, 130, 130, 130, 130, 144), ((), (), frozenset(), None)),
        # Empty board: three empty cells must not count as a line.
        ((_,) * 9, (200, 0, 0, 0, 0, 0, 0, 824), ((0, 7), (0, 7), frozenset(), None)),
        # Double win, both lines flagged.
        (DOUBLE_WIN, (512, 0, 0, 512, 0, 0, 0, 0), ((0, 3), (), frozenset({0, 3}), X)),
        # Double win, only one line flagged: still a win.
        (DOUBLE_WIN, (512, 0, 0, 100, 0, 0, 0, 412), ((0, 7), (7,), frozenset({0}), X)),
        # A count equal to the cutoff is flagged.
        (TOP_ROW, (153, 0, 0, 0, 0, 0, 0, 871), ((0, 7), (7,), frozenset({0}), X)),
        # O wins the middle column.
        ((X, O, X,
          _, O, X,
          _, O, _), (0, 0, 0, 0, 1024, 0, 0, 0), ((4,), (), frozenset({4}), O)),
    ],
)
def test_flag_and_check(board, counts, expected):
    assert flag_and_check(board, counts, 153) == expected


def test_flag_and_check_only_reads_flagged_lines():
    # Cells outside the flagged line are never read, so they can hold anything.
    class Board(tuple):
        def __getitem__(self, i):
            assert i in (0, 1, 2), f"read cell {i} outside flagged line 0"
            return super().__getitem__(i)

    board = Board(TOP_ROW)
    assert flag_and_check(board, (1024, 0, 0, 0, 0, 0, 0, 0), 153)[3] == X


# --- run_search ---

def test_run_search_counts():
    run = run_search(TOP_ROW, player=X, backend=IdealBackend(seed=SEED))
    assert run.shots == DEFAULT_SHOTS
    assert len(run.counts) == NUM_LINES
    assert sum(run.counts) == DEFAULT_SHOTS
    assert max(range(NUM_LINES), key=run.counts.__getitem__) == 0
    assert all(isinstance(c, int) for c in run.counts)


def test_run_search_is_repeatable_with_seed():
    a = run_search(TOP_ROW, player=X, shots=256, backend=IdealBackend(seed=SEED))
    b = run_search(TOP_ROW, player=X, shots=256, backend=IdealBackend(seed=SEED))
    assert a.counts == b.counts


def test_run_search_rejects_unknown_variant():
    with pytest.raises(ValueError):
        run_search(TOP_ROW, variant="nope")


# --- detect_winner, legacy 22-qubit full variant, end to end on the ideal simulator ---

def _check_result(result, board):
    winners = _winners(board)
    # Never a wrong line, whatever the counts were.
    assert result.lines <= winners
    assert set(result.candidates) == set(result.lines) | set(result.rejected)
    assert not set(result.lines) & set(result.rejected)
    assert result.checks == len(result.candidates)
    assert result.board == tuple(board)
    assert result.cutoffs == (binomial_cutoff(result.run.shots, result.budget),) * NUM_LINES


@pytest.mark.slow
@pytest.mark.parametrize("board", SINGLE_WIN_SAMPLE, ids=_board_id)
def test_detect_single_win(board):
    result = detect_winner(board, variant="full", backend=IdealBackend(seed=SEED))
    _check_result(result, board)
    (line,) = _winners(board)
    assert result.lines == {line}
    assert result.winner == board[WIN_LINES[line][0]]


@pytest.mark.slow
@pytest.mark.parametrize("board", NO_WIN_SAMPLE, ids=_board_id)
def test_detect_no_win(board):
    result = detect_winner(board, variant="full", backend=IdealBackend(seed=SEED))
    _check_result(result, board)
    assert result.winner is None
    assert result.lines == frozenset()


@pytest.mark.slow
@pytest.mark.parametrize("board", DOUBLE_WINS, ids=_board_id)
def test_detect_double_win(board):
    result = detect_winner(board, variant="full", backend=IdealBackend(seed=SEED))
    _check_result(result, board)
    assert result.winner == X
    assert result.lines == _winners(board)


def test_detect_winner_returns_tuple_board():
    result = detect_winner(list(TOP_ROW), player=X, shots=256, backend=IdealBackend(seed=SEED))
    assert result.board == TOP_ROW
    assert result.variant == "player"
    assert result.budget == DEFAULT_BUDGET


@pytest.mark.parametrize("kwargs", [{"shots": 0}, {"budget": 0}])
def test_detect_winner_checks_input_before_running(kwargs, monkeypatch):
    def fail(*args, **kw):
        raise AssertionError("circuit ran before input was checked")

    monkeypatch.setattr("qttt.detector.run_searches", fail)
    with pytest.raises(ValueError):
        detect_winner(TOP_ROW, **kwargs)


# --- detect_winner, default 12-qubit player variant ---

def _last_player(board):
    return X if board.count(X) > board.count(O) else O


@pytest.mark.parametrize("board", SINGLE_WIN_SAMPLE + NO_WIN_SAMPLE + DOUBLE_WINS, ids=_board_id)
def test_detect_player_variant(board):
    player = _last_player(board)
    result = detect_winner(board, player=player, backend=IdealBackend(seed=SEED))
    _check_result(result, board)
    assert result.variant == "player"
    assert result.lines == _winners(board)
    assert result.winner == (player if result.lines else None)
    assert result.run.circuit.num_qubits == 12


def test_detect_player_variant_needs_player():
    with pytest.raises(ValueError):
        detect_winner(TOP_ROW)


# --- profile modes: calibration inside the same job ---

class _CountingBackend(IdealBackend):
    """Ideal backend that records each job's size."""

    def __init__(self, seed=None):
        super().__init__(seed)
        self.jobs = []

    def run(self, circuits, shots, on_submit=None):
        self.jobs.append(len(circuits))
        return super().run(circuits, shots, on_submit)


def test_auto_on_ideal_backend_does_not_calibrate():
    backend = _CountingBackend(seed=SEED)
    result = detect_winner(TOP_ROW, player=X, shots=256, backend=backend)
    assert result.profile.source == "ideal"
    assert backend.jobs == [1]


def test_calibrate_runs_in_the_same_job():
    backend = _CountingBackend(seed=SEED)
    cases = [(TOP_ROW, X), (DOUBLE_WIN, X)]
    results = detect_winners(cases, shots=2048, backend=backend, profile="calibrate")
    assert backend.jobs == [CALIBRATION_CIRCUITS + len(cases)]  # readout circuits, known boards, the boards
    profile = results[0].profile
    assert profile.source == "calibration:ideal"
    assert profile.line_qubits == results[0].run.metrics.line_qubits
    assert profile.epsilon == pytest.approx(0.0, abs=0.05)  # no noise to find
    assert [r.winner for r in results] == [X, X]
    assert results[0].cutoffs == profile.cutoffs(2048, DEFAULT_BUDGET)


def test_published_profile_on_ideal_backend_is_ideal():
    result = detect_winner(TOP_ROW, player=X, shots=256, backend=IdealBackend(seed=SEED), profile="published")
    assert result.profile.epsilon == 0.0
    assert result.cutoffs == (binomial_cutoff(256, DEFAULT_BUDGET),) * NUM_LINES


def test_unknown_profile_mode_is_refused_before_running(monkeypatch):
    monkeypatch.setattr("qttt.backends.Backend.run", lambda *a, **k: pytest.fail("ran"))
    with pytest.raises(ValueError, match="Unknown profile"):
        detect_winner(TOP_ROW, player=X, profile="best")


def test_calibration_is_stored_and_linked(tmp_path):
    from qttt.results import RunStore

    store = RunStore(tmp_path)
    detect_winners([(TOP_ROW, X), (DOUBLE_WIN, X)], shots=256, backend=IdealBackend(seed=SEED),
                   profile="calibrate", store=store)
    (calibration,) = store.load("calibration")
    detections = store.load("detection")
    assert len(detections) == 2
    assert {d["calibration_id"] for d in detections} == {calibration["id"]}
    assert detections[0]["profile"]["source"] == "calibration:ideal"


@pytest.mark.slow
def test_calibrate_with_legacy_full_variant():
    result = detect_winner(TOP_ROW, variant="full", shots=512, backend=IdealBackend(seed=SEED), profile="calibrate")
    assert result.winner == X
    assert result.profile.epsilon == pytest.approx(0.0, abs=0.05)


# --- a job that ran is never lost ---

class _StoppedBackend(IdealBackend):
    """Ideal backend whose job is submitted but never waited for, like a kernel stopped mid-queue.

    fetch returns the counts the job would have produced.
    """

    def _sample(self, circuits, shots, on_submit):
        self.output = IdealBackend(self.seed).run(circuits, shots)
        on_submit("job-1")
        raise KeyboardInterrupt

    def fetch(self, job_id):
        assert job_id == "job-1"
        return self.output


def test_submitted_job_is_recorded_before_its_results(tmp_path):
    store = RunStore(tmp_path)
    backend = _StoppedBackend(seed=SEED)
    with pytest.raises(KeyboardInterrupt):
        detect_winners([(TOP_ROW, X)], shots=256, backend=backend, store=store, tag="t")
    (submitted,) = store.load("job")
    assert submitted["status"] == "submitted" and submitted["job_id"] == "job-1"
    assert submitted["cases"] == [{"board": "XXXOO    ", "player": X}]
    assert store.load("detection") == []

    (result,) = resume_job("job-1", backend, store)
    assert result.winner == X and result.run.counts == backend.output.counts[0]
    assert [r["status"] for r in store.load("job")] == ["submitted", "done"]
    (detection,) = store.load("detection")
    assert detection["job_record"] == submitted["id"] and detection["tag"] == "t"


def test_counts_are_saved_before_a_crash_and_the_job_can_be_finished(tmp_path, monkeypatch):
    store = RunStore(tmp_path)
    backend = _CountingBackend(seed=SEED)
    cases = [(TOP_ROW, X), (DOUBLE_WIN, X)]
    expected = detect_winners(cases, shots=256, backend=IdealBackend(seed=SEED), profile="calibrate")

    def crash(*args):
        raise RuntimeError("bug after the run")
    monkeypatch.setattr("qttt.detector.flag_and_check", crash)
    with pytest.raises(RuntimeError) as info:
        detect_winners(cases, shots=256, backend=backend, profile="calibrate", store=store)
    submitted, done = store.job(store.load("job")[0]["id"])
    assert "resume_job" in "".join(info.value.__notes__)
    assert done["counts"][-2:] == [list(r.run.counts) for r in expected]
    monkeypatch.undo()

    results = resume_job(submitted["id"], IdealBackend(seed=SEED), store)
    assert backend.jobs == [CALIBRATION_CIRCUITS + 2]  # resuming runs nothing
    assert [r.lines for r in results] == [r.lines for r in expected]
    assert results[0].cutoffs == expected[0].cutoffs
    (calibration,) = store.load("calibration")
    assert {d["calibration_id"] for d in store.load("detection")} == {calibration["id"]}
    with pytest.raises(ValueError, match="already finished"):
        resume_job(submitted["id"], IdealBackend(seed=SEED), store)


def test_resume_refuses_another_backend(tmp_path):
    store = RunStore(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        detect_winners([(TOP_ROW, X)], shots=64, backend=_StoppedBackend(seed=SEED), store=store)

    class Other(IdealBackend):
        name = "other"
    with pytest.raises(ValueError, match="ran on ideal"):
        resume_job("job-1", Other(), store)


def test_a_reused_profile_links_to_its_calibration(tmp_path):
    store = RunStore(tmp_path)
    (first,) = detect_winners([(TOP_ROW, X)], shots=256, backend=IdealBackend(seed=SEED), profile="calibrate",
                              store=store)
    (calibration,) = store.load("calibration")
    assert first.profile.calibration_id == calibration["id"]
    detect_winner(DOUBLE_WIN, player=X, shots=256, backend=IdealBackend(seed=SEED), profile=first.profile, store=store)
    assert [d["calibration_id"] for d in store.load("detection")] == [calibration["id"]] * 2
    assert store.load("calibration") == [calibration]  # reusing the profile runs no new calibration
