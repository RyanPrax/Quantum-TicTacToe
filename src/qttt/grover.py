"""Grover diffuser and search-circuit builder."""

from collections.abc import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister, QuantumRegister

from qttt.encoding import encode_full_board
from qttt.oracles import full_oracle

VARIANTS = ("full",)


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


def build_search_circuit(
    variant: str,
    board: Sequence[str],
    player: str | None = None,
    iterations: int = 2,
) -> QuantumCircuit:
    """Build a Grover search over the 8 winning lines for the given board.

    Registers are named "board", "line" and "phase", and the line register is
    measured into a classical register named "line_cl". The "full" variant
    checks both players, so it ignores player.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANTS}.")
    if iterations < 0:
        raise ValueError("Iterations must be non-negative.")

    board_reg = QuantumRegister(18, name="board")
    line_reg = QuantumRegister(3, name="line")
    phase_reg = QuantumRegister(1, name="phase")
    line_cl = ClassicalRegister(3, name="line_cl")
    qc = QuantumCircuit(board_reg, line_reg, phase_reg, line_cl, name=f"grover_{variant}")

    encode_full_board(qc, board_reg, board)

    # Phase qubit in |-> for phase kickback; line register in uniform superposition.
    qc.x(phase_reg[0])
    qc.h(phase_reg[0])
    qc.h(line_reg)
    qc.barrier(label="Setup")

    for i in range(iterations):
        full_oracle(qc, board_reg, line_reg, phase_reg[0])
        qc.barrier(label=f"Oracle_{i + 1}")
        diffuser(qc, line_reg)
        qc.barrier(label=f"Diffuser_{i + 1}")

    qc.measure(line_reg, line_cl)
    return qc
