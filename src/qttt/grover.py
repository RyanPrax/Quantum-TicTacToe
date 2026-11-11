"""Grover diffuser and search-circuit builder."""

from collections.abc import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister, QuantumRegister

from qttt.encoding import encode_full_board, encode_player_board
from qttt.oracles import full_oracle, player_oracle

# "player" (13 qubits) is the main version; "full" (22 qubits) is the legacy class version.
VARIANTS = ("full", "player")


def diffuser(qc: QuantumCircuit, qubits: QuantumRegister) -> None:
    """Apply the diffuser (inversion about the mean) to the given qubits."""
    n = len(qubits)
    if n == 0:
        return
    qc.h(qubits)
    qc.x(qubits)
    if n > 1:
        # Multi-controlled Z, built from an MCX targeting the last qubit.
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    else:
        qc.z(qubits[0])
    qc.x(qubits)
    qc.h(qubits)


def board_qubit_count(variant: str) -> int:
    """Size of the board register: 18 for "full", 9 for "player"."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANTS}.")
    return 18 if variant == "full" else 9


def board_flips(variant: str, board: Sequence[str], player: str | None = None) -> list[int]:
    """Indices of the board-register qubits the encoder sets to |1> for this board.

    The board register is the first register of the search circuit, so these
    are also the board's qubit indices in build_search_circuit().
    """
    board_reg = QuantumRegister(board_qubit_count(variant), name="board")
    qc = QuantumCircuit(board_reg)
    if variant == "full":
        encode_full_board(qc, board_reg, board)
    else:
        encode_player_board(qc, board_reg, board, player)
    return [qc.find_bit(inst.qubits[0]).index for inst in qc.data]


def build_search_circuit(
    variant: str,
    board: Sequence[str],
    player: str | None = None,
) -> QuantumCircuit:
    """Build a one-iteration Grover search over the 8 winning lines for the given board.

    Registers are named "board", "line" and "phase", and the line register is
    measured into a classical register named "line_cl". The "player" variant
    (13 qubits) checks only the given player, who must be the one who just moved.
    The legacy "full" variant (22 qubits, the class version) checks both
    players, so it ignores player.
    """
    board_reg = QuantumRegister(board_qubit_count(variant), name="board")
    line_reg = QuantumRegister(3, name="line")
    phase_reg = QuantumRegister(1, name="phase")
    line_cl = ClassicalRegister(3, name="line_cl")
    qc = QuantumCircuit(board_reg, line_reg, phase_reg, line_cl, name=f"grover_{variant}")

    if variant == "full":
        encode_full_board(qc, board_reg, board)
    else:
        encode_player_board(qc, board_reg, board, player)

    # Phase qubit in |-> for phase kickback; line register in uniform superposition.
    qc.x(phase_reg[0])
    qc.h(phase_reg[0])
    qc.h(line_reg)
    qc.barrier(label="Setup")

    if variant == "full":
        full_oracle(qc, board_reg, line_reg, phase_reg[0])
    else:
        player_oracle(qc, board_reg, line_reg, phase_reg[0])
    qc.barrier(label="Oracle")
    diffuser(qc, line_reg)
    qc.barrier(label="Diffuser")

    qc.measure(line_reg, line_cl)
    return qc
