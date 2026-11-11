"""Tests for the winner detector.

Cutoff and flag-and-check tests are pure Python. End-to-end tests run the
22-qubit circuit on the ideal simulator (about a second per board) with a fixed
seed, so their counts are the same on every run.
"""

import dataclasses
import math
import random

import pytest
from qiskit import QuantumCircuit
from qiskit_aer.primitives import SamplerV2

from qttt.board import EMPTY, WIN_LINES, O, X, boards_by_win_count, classical_winning_lines
from qttt.detector import (
    DEFAULT_BUDGET,
    DEFAULT_SHOTS,
    NUM_LINES,
    DetectionResult,
    SearchRun,
    _binomial_log_pmf,
    _binomial_tail,
    _flag_and_check,
    binomial_cutoff,
    detect_winner,
    run_search,
)

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
    while NUM_LINES * _binomial_tail(k, shots, q0) > budget:
        k += 1
    return k


# --- Binomial helpers ---

def test_log_pmf_matches_direct_formula():
    for j in range(5):
        direct = math.comb(4, j) * (1 / 8) ** j * (7 / 8) ** (4 - j)
        assert math.exp(_binomial_log_pmf(j, 4, 1 / 8)) == pytest.approx(direct)


@pytest.mark.parametrize("n", [1, 10, 1024])
def test_pmf_sums_to_one(n):
    total = sum(math.exp(_binomial_log_pmf(j, n, 1 / 8)) for j in range(n + 1))
    assert total == pytest.approx(1.0)


def test_tail_small_example():
    # 4 shots: P(K >= 2) = 0.072 + 0.0068 + 0.00024
    assert _binomial_tail(2, 4, 1 / 8) == pytest.approx(0.07886, abs=1e-5)


def test_tail_edges():
    assert _binomial_tail(0, 10, 0.3) == 1.0
    assert _binomial_tail(-1, 10, 0.3) == 1.0
    assert _binomial_tail(11, 10, 0.3) == 0.0


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
    assert NUM_LINES * _binomial_tail(k, shots, 1 / 8) <= DEFAULT_BUDGET
    assert NUM_LINES * _binomial_tail(k - 1, shots, 1 / 8) > DEFAULT_BUDGET


@pytest.mark.parametrize("q0", [0.05, 1 / 8, 0.2])
@pytest.mark.parametrize("budget", [0.01, 0.1, 1.0])
def test_cutoff_matches_reference(budget, q0):
    for shots in range(1, 151):
        assert binomial_cutoff(shots, budget, q0) == _reference_cutoff(shots, budget, q0)


def test_defaults():
    assert binomial_cutoff() == binomial_cutoff(1024, 0.1, 1 / 8)


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
        board=TOP_ROW, variant="full", winner=X, lines=frozenset({0}),
        candidates=(0, 7), rejected=(7,), cutoff=153, budget=0.1, run=run,
    )
    assert result.probabilities == run.probabilities
    assert result.checks == 2


def test_results_are_frozen():
    run = _run()
    with pytest.raises(dataclasses.FrozenInstanceError):
        run.shots = 1
    result = DetectionResult(TOP_ROW, "full", None, frozenset(), (), (), 153, 0.1, run)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.winner = X


# --- _flag_and_check (made-up counts, no circuit) ---

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
    assert _flag_and_check(board, counts, 153) == expected


def test_flag_and_check_only_reads_flagged_lines():
    # Cells outside the flagged line are never read, so they can hold anything.
    class Board(tuple):
        def __getitem__(self, i):
            assert i in (0, 1, 2), f"read cell {i} outside flagged line 0"
            return super().__getitem__(i)

    board = Board(TOP_ROW)
    assert _flag_and_check(board, (1024, 0, 0, 0, 0, 0, 0, 0), 153)[3] == X


# --- run_search ---

def test_run_search_counts():
    run = run_search(TOP_ROW, player=X, sampler=SamplerV2(seed=SEED))
    assert run.shots == DEFAULT_SHOTS
    assert len(run.counts) == NUM_LINES
    assert sum(run.counts) == DEFAULT_SHOTS
    assert max(range(NUM_LINES), key=run.counts.__getitem__) == 0
    assert all(isinstance(c, int) for c in run.counts)


def test_run_search_is_repeatable_with_seed():
    a = run_search(TOP_ROW, player=X, shots=256, sampler=SamplerV2(seed=SEED))
    b = run_search(TOP_ROW, player=X, shots=256, sampler=SamplerV2(seed=SEED))
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
    assert result.cutoff == binomial_cutoff(result.run.shots, result.budget)


@pytest.mark.parametrize("board", SINGLE_WIN_SAMPLE, ids=_board_id)
def test_detect_single_win(board):
    result = detect_winner(board, variant="full", sampler=SamplerV2(seed=SEED))
    _check_result(result, board)
    (line,) = _winners(board)
    assert result.lines == {line}
    assert result.winner == board[WIN_LINES[line][0]]


@pytest.mark.parametrize("board", NO_WIN_SAMPLE, ids=_board_id)
def test_detect_no_win(board):
    result = detect_winner(board, variant="full", sampler=SamplerV2(seed=SEED))
    _check_result(result, board)
    assert result.winner is None
    assert result.lines == frozenset()


@pytest.mark.parametrize("board", DOUBLE_WINS, ids=_board_id)
def test_detect_double_win(board):
    result = detect_winner(board, variant="full", sampler=SamplerV2(seed=SEED))
    _check_result(result, board)
    assert result.winner == X
    assert result.lines == _winners(board)


def test_detect_winner_returns_tuple_board():
    result = detect_winner(list(TOP_ROW), player=X, shots=256, sampler=SamplerV2(seed=SEED))
    assert result.board == TOP_ROW
    assert result.variant == "player"
    assert result.budget == DEFAULT_BUDGET


@pytest.mark.parametrize("kwargs", [{"shots": 0}, {"budget": 0}])
def test_detect_winner_checks_input_before_running(kwargs, monkeypatch):
    def fail(*args, **kw):
        raise AssertionError("circuit ran before input was checked")

    monkeypatch.setattr("qttt.detector.run_search", fail)
    with pytest.raises(ValueError):
        detect_winner(TOP_ROW, **kwargs)


# --- detect_winner, default 13-qubit player variant ---

def _last_player(board):
    return X if board.count(X) > board.count(O) else O


@pytest.mark.parametrize("board", SINGLE_WIN_SAMPLE + NO_WIN_SAMPLE + DOUBLE_WINS, ids=_board_id)
def test_detect_player_variant(board):
    player = _last_player(board)
    result = detect_winner(board, player=player, sampler=SamplerV2(seed=SEED))
    _check_result(result, board)
    assert result.variant == "player"
    assert result.lines == _winners(board)
    assert result.winner == (player if result.lines else None)
    assert result.run.circuit.num_qubits == 13


def test_detect_player_variant_needs_player():
    with pytest.raises(ValueError):
        detect_winner(TOP_ROW)
