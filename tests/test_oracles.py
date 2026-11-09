"""Check each oracle against the classical checker.

The 22-qubit simulation takes about a second per board, so the full oracle is
tested on a fixed sample of boards plus every double-win board.
"""

import random

import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import QuantumRegister
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from qttt.board import EMPTY, O, X, boards_by_win_count, classical_winning_lines
from qttt.encoding import (
    bitstring_to_line_index,
    cell_bits,
    encode_full_board,
    line_index_to_bitstring,
)
from qttt.grover import build_search_circuit

SAMPLE_SIZE = 10
SEED = 4903

_SIMULATOR = AerSimulator(method="statevector")


def line_probabilities(variant, board, player=None):
    """Exact probability of measuring each line index 0-7."""
    qc = build_search_circuit(variant, board, player)
    line_reg = qc.qregs[1]
    qc = qc.remove_final_measurements(inplace=False)
    qc.save_probabilities(line_reg)
    result = _SIMULATOR.run(transpile(qc, _SIMULATOR)).result()
    return list(result.data()["probabilities"])


_GROUPS = boards_by_win_count()
_rng = random.Random(SEED)
NO_WIN_SAMPLE = _rng.sample(_GROUPS[0], SAMPLE_SIZE)
SINGLE_WIN_SAMPLE = _rng.sample(_GROUPS[1], SAMPLE_SIZE)
DOUBLE_WINS = _GROUPS[2]


def _winners(board):
    return classical_winning_lines(board, X) | classical_winning_lines(board, O)


# --- Encoding helpers ---

@pytest.mark.parametrize("index", range(8))
def test_line_index_bitstring_round_trip(index):
    bits = line_index_to_bitstring(index)
    assert len(bits) == 3
    assert bitstring_to_line_index(bits) == index


def test_bitstring_reads_as_binary():
    # Qiskit prints qubit 0 on the right, so index 1 sets only line_reg[0].
    assert line_index_to_bitstring(1) == "001"
    assert line_index_to_bitstring(6) == "110"


@pytest.mark.parametrize("bad", ["", "01", "0102", "abc"])
def test_bitstring_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        bitstring_to_line_index(bad)


def test_encode_full_board_marks():
    board = [X, O, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY]
    board_reg = QuantumRegister(18, name="board")
    qc = QuantumCircuit(board_reg)
    encode_full_board(qc, board_reg, board)

    x_q0, x_q1 = cell_bits(0)
    o_q0, o_q1 = cell_bits(1)
    expected = (1 << x_q1) | (1 << o_q0)  # X = |10>, O = |01> as |q1 q0>
    probs = Statevector(qc).probabilities()
    assert probs[expected] == pytest.approx(1.0)


# --- Full (22-qubit) oracle ---
# One Grover iteration over 8 lines: M = 1 puts 25/32 on the winner and 1/32 on
# each other line; M = 2 puts 1/2 on each winner and 0 elsewhere.

@pytest.mark.parametrize("board", SINGLE_WIN_SAMPLE, ids=lambda b: "".join(b).replace(" ", "."))
def test_full_oracle_single_win(board):
    probs = line_probabilities("full", board)
    (winner,) = _winners(board)
    for i in range(8):
        assert probs[i] == pytest.approx(25 / 32 if i == winner else 1 / 32, abs=1e-9)


@pytest.mark.parametrize("board", NO_WIN_SAMPLE, ids=lambda b: "".join(b).replace(" ", "."))
def test_full_oracle_no_win(board):
    probs = line_probabilities("full", board)
    assert probs == pytest.approx([1 / 8] * 8)


@pytest.mark.parametrize("board", DOUBLE_WINS, ids=lambda b: "".join(b).replace(" ", "."))
def test_full_oracle_double_win(board):
    probs = line_probabilities("full", board)
    winners = _winners(board)
    for i in range(8):
        assert probs[i] == pytest.approx(0.5 if i in winners else 0.0, abs=1e-9)
