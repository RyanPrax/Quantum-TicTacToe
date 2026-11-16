"""Check each oracle against the classical checker.

The 22-qubit simulation takes about a second per board, so the full oracle is
tested on a fixed sample of boards plus every double-win board. The 12-qubit
player oracle is fast enough to test on every board the game checks.
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
    encode_player_board,
    line_index_to_bitstring,
)
from qttt.grover import build_phase_kickback_circuit, build_search_circuit

SAMPLE_SIZE = 10
SEED = 4903

_SIMULATOR = AerSimulator(method="statevector")


def line_probabilities(variant, board, player=None, ancilla=False):
    """Exact probability of measuring each line index 0-7."""
    return circuit_line_probabilities(build_search_circuit(variant, board, player, ancilla))


def circuit_line_probabilities(qc):
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


def _last_player(board):
    return X if board.count(X) > board.count(O) else O


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


@pytest.mark.parametrize(("player", "expected"), [(X, 0b10101), (O, 0b01010)])
def test_encode_player_board_marks(player, expected):
    # X owns cells 0, 2, 4 and O owns cells 1, 3; qubit c is |1> when the player owns cell c.
    board = [X, O, X, O, X, EMPTY, EMPTY, EMPTY, EMPTY]
    board_reg = QuantumRegister(9, name="board")
    qc = QuantumCircuit(board_reg)
    encode_player_board(qc, board_reg, board, player)
    assert Statevector(qc).probabilities()[expected] == pytest.approx(1.0)


@pytest.mark.parametrize(("board", "player"), [([EMPTY] * 9, EMPTY), ([EMPTY] * 9, None), ([EMPTY] * 8, X)])
def test_encode_player_board_rejects_bad_input(board, player):
    board_reg = QuantumRegister(9, name="board")
    with pytest.raises(ValueError):
        encode_player_board(QuantumCircuit(board_reg), board_reg, board, player)


# --- Full (22-qubit) oracle ---
# One Grover iteration over 8 lines: M = 1 puts 25/32 on the winner and 1/32 on
# each other line; M = 2 puts 1/2 on each winner and 0 elsewhere.

@pytest.mark.slow
@pytest.mark.parametrize("board", SINGLE_WIN_SAMPLE, ids=lambda b: "".join(b).replace(" ", "."))
def test_full_oracle_single_win(board):
    probs = line_probabilities("full", board)
    (winner,) = _winners(board)
    for i in range(8):
        assert probs[i] == pytest.approx(25 / 32 if i == winner else 1 / 32, abs=1e-9)


@pytest.mark.slow
@pytest.mark.parametrize("board", NO_WIN_SAMPLE, ids=lambda b: "".join(b).replace(" ", "."))
def test_full_oracle_no_win(board):
    probs = line_probabilities("full", board)
    assert probs == pytest.approx([1 / 8] * 8)


@pytest.mark.slow
@pytest.mark.parametrize("board", DOUBLE_WINS, ids=lambda b: "".join(b).replace(" ", "."))
def test_full_oracle_double_win(board):
    probs = line_probabilities("full", board)
    winners = _winners(board)
    for i in range(8):
        assert probs[i] == pytest.approx(0.5 if i in winners else 0.0, abs=1e-9)


# --- Player (12-qubit) oracle, every board from turn 5 on ---

@pytest.fixture(scope="module")
def player_probabilities():
    """Exact line probabilities for every board, keyed by (board, player).

    Each board runs for the player who just moved (the real case) and for the
    opponent (who never has a line). Aer runs mcx directly, so the circuits
    skip transpiling and run as one batch.
    """
    keys = [(board, player) for group in _GROUPS.values() for board in group
            for player in (_last_player(board), _other(board))]
    circuits = []
    for board, player in keys:
        qc = build_search_circuit("player", board, player).remove_final_measurements(inplace=False)
        qc.save_probabilities(qc.qregs[1])
        circuits.append(qc)
    result = _SIMULATOR.run(circuits).result()
    return {key: list(result.data(i)["probabilities"]) for i, key in enumerate(keys)}


def _other(board):
    return O if _last_player(board) == X else X


def _expected(winners):
    if not winners:
        return [1 / 8] * 8
    if len(winners) == 1:
        return [25 / 32 if i in winners else 1 / 32 for i in range(8)]
    return [0.5 if i in winners else 0.0 for i in range(8)]


@pytest.mark.slow
@pytest.mark.parametrize("wins", [0, 1, 2], ids=["no_win", "single_win", "double_win"])
def test_player_oracle_every_board(player_probabilities, wins):
    boards = _GROUPS[wins]
    wrong = [
        board for board in boards
        if player_probabilities[(board, _last_player(board))]
        != pytest.approx(_expected(_winners(board)), abs=1e-9)
    ]
    assert not wrong, f"{len(wrong)} of {len(boards)} boards wrong, e.g. {wrong[:3]}"


@pytest.mark.slow
def test_player_oracle_opponent_never_wins(player_probabilities):
    # The opponent of the last mover owns no line, so the search stays uniform.
    wrong = [
        board for group in _GROUPS.values() for board in group
        if player_probabilities[(board, _other(board))] != pytest.approx([1 / 8] * 8)
    ]
    assert not wrong, f"{len(wrong)} boards wrong, e.g. {wrong[:3]}"


def test_player_variant_size_and_gates():
    qc = build_search_circuit("player", SINGLE_WIN_SAMPLE[0], _last_player(SINGLE_WIN_SAMPLE[0]))
    assert qc.num_qubits == 12
    assert [r.name for r in qc.qregs] == ["board", "line"]
    mcx = [inst for inst in qc.data if inst.operation.name == "mcx"]
    # 8 oracle MCZ gates on 6 qubits, each an MCX with 5 controls; the diffuser's 2-control MCX is a ccx.
    assert [len(inst.qubits) - 1 for inst in mcx] == [5] * 8


def test_player_variant_with_ancilla_size_and_gates():
    qc = build_search_circuit("player", SINGLE_WIN_SAMPLE[0], _last_player(SINGLE_WIN_SAMPLE[0]), ancilla=True)
    assert qc.num_qubits == 13
    assert [r.name for r in qc.qregs] == ["board", "line", "ancilla"]
    assert "mcx" not in qc.count_ops()  # every oracle MCX is already synthesized


ANCILLA_BOARDS = NO_WIN_SAMPLE + SINGLE_WIN_SAMPLE + DOUBLE_WINS


@pytest.mark.parametrize("board", ANCILLA_BOARDS, ids=["".join(b).replace(" ", "_") for b in ANCILLA_BOARDS])
def test_player_oracle_with_ancilla(board):
    probs = line_probabilities("player", board, _last_player(board), ancilla=True)
    assert probs == pytest.approx(_expected(_winners(board)), abs=1e-9)


def test_ancilla_is_returned_clean():
    board = SINGLE_WIN_SAMPLE[0]
    qc = build_search_circuit("player", board, _last_player(board), ancilla=True)
    qc = qc.remove_final_measurements(inplace=False)
    ancilla = qc.find_bit(qc.qregs[2][0]).index
    assert Statevector(qc).probabilities([ancilla])[1] == pytest.approx(0.0, abs=1e-9)


def test_full_variant_has_no_ancilla_option():
    with pytest.raises(ValueError, match="ancilla"):
        build_search_circuit("full", SINGLE_WIN_SAMPLE[0], ancilla=True)


def test_player_variant_needs_player():
    with pytest.raises(ValueError):
        build_search_circuit("player", SINGLE_WIN_SAMPLE[0])


@pytest.mark.parametrize("variant, ancilla, exposed", [
    ("player", False, [9, 10, 11]),
    ("player", True, [9, 10, 11, 12]),
    ("full", False, [18, 19, 20, 21]),
])
def test_search_circuit_names_its_qubits_in_superposition(variant, ancilla, exposed):
    board = SINGLE_WIN_SAMPLE[0]
    qc = build_search_circuit(variant, board, _last_player(board) if variant == "player" else None, ancilla)
    assert qc.metadata["exposed_qubits"] == exposed


@pytest.mark.parametrize("board", [NO_WIN_SAMPLE[0], SINGLE_WIN_SAMPLE[0], DOUBLE_WINS[0]], ids=["none", "one", "two"])
def test_phase_kickback_circuit_matches_player_variant(board):
    player = _last_player(board)
    qc = build_phase_kickback_circuit(board, player)
    assert qc.num_qubits == 13
    assert qc.count_ops()["mcx"] == 8
    assert circuit_line_probabilities(qc) == pytest.approx(line_probabilities("player", board, player), abs=1e-9)
