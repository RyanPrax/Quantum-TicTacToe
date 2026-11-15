"""Grover diffuser and search-circuit builder."""

from collections.abc import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister, QuantumRegister

from qttt.encoding import encode_full_board, encode_player_board
from qttt.oracles import full_oracle, mcz, player_oracle

# "player" (12 qubits) is the main version; "full" (22 qubits) is the legacy class version.
VARIANTS = ("full", "player")


def diffuser(qc: QuantumCircuit, qubits: QuantumRegister) -> None:
    """Apply the diffuser (inversion about the mean) to the given qubits."""
    n = len(qubits)
    if n == 0:
        return
    qc.h(qubits)
    qc.x(qubits)
    if n > 1:
        mcz(qc, list(qubits))
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
    ancilla: bool = False,
) -> QuantumCircuit:
    """Build a one-iteration Grover search over the 8 winning lines for the given board.

    The line register is named "line" and is measured into a classical register
    named "line_cl"; the board register "board" comes first. The "player" variant
    (12 qubits) checks only the given player, who must be the one who just moved;
    ancilla=True adds a 13th qubit, "ancilla", that lets each oracle gate be
    built with fewer two-qubit gates. The legacy "full" variant (22 qubits, the
    class version) checks both players with a "phase" qubit, so it ignores player.
    """
    if ancilla and variant == "full":
        raise ValueError("ancilla is only available for the player variant")
    board_reg = QuantumRegister(board_qubit_count(variant), name="board")
    line_reg = QuantumRegister(3, name="line")
    line_cl = ClassicalRegister(3, name="line_cl")

    if variant == "full":
        phase_reg = QuantumRegister(1, name="phase")
        qc = QuantumCircuit(board_reg, line_reg, phase_reg, line_cl, name="grover_full")
        encode_full_board(qc, board_reg, board)
        # Phase qubit in |-> for phase kickback.
        qc.x(phase_reg[0])
        qc.h(phase_reg[0])
    else:
        ancilla_reg = [QuantumRegister(1, name="ancilla")] if ancilla else []
        qc = QuantumCircuit(board_reg, line_reg, *ancilla_reg, line_cl, name="grover_player")
        encode_player_board(qc, board_reg, board, player)
    qc.h(line_reg)
    qc.barrier(label="Setup")

    if variant == "full":
        full_oracle(qc, board_reg, line_reg, phase_reg[0])
    else:
        player_oracle(qc, board_reg, line_reg, ancilla_reg[0][0] if ancilla else None)
    qc.barrier(label="Oracle")
    diffuser(qc, line_reg)
    qc.barrier(label="Diffuser")

    qc.measure(line_reg, line_cl)
    # Logical qubits held in superposition the whole time; the board qubits only hold 0 or 1 as controls.
    exposed = list(line_reg) + (list(phase_reg) if variant == "full" else ancilla_reg[0][:] if ancilla else [])
    qc.metadata = {"ancilla": ancilla, "exposed_qubits": [qc.find_bit(q).index for q in exposed]}
    return qc
