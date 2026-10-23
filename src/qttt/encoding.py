"""Load a classical board into qubits, and convert line indices to and from bitstrings."""

from collections.abc import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister

from qttt.board import O, X


def cell_bits(cell: int) -> tuple[int, int]:
    """Return the board-register indices (q0, q1) used by a cell (0-8) in the 2-qubit encoding."""
    if not 0 <= cell <= 8:
        raise ValueError("Cell index must be between 0 and 8.")
    return 2 * cell, 2 * cell + 1


def encode_full_board(qc: QuantumCircuit, board_reg: QuantumRegister, board: Sequence[str]) -> None:
    """Load both players' marks into an 18-qubit register, 2 qubits per cell.

    Written as |q1 q0> (Qiskit order, q0 on the right):
    empty = |00>, X = |10>, O = |01>.
    """
    if len(board) != 9:
        raise ValueError("Board must have exactly 9 cells.")
    for cell, mark in enumerate(board):
        q0, q1 = cell_bits(cell)
        if mark == X:
            qc.x(board_reg[q1])
        elif mark == O:
            qc.x(board_reg[q0])


def line_index_to_bitstring(index: int) -> str:
    """Return the 3-bit string Qiskit reports when the line register measures this line index.

    Qiskit prints qubit 0 on the right, so line_reg[0] is the last character
    and the string is the line index written in binary.
    """
    if not 0 <= index <= 7:
        raise ValueError("Line index must be between 0 and 7.")
    return format(index, "03b")


def bitstring_to_line_index(bits: str) -> int:
    """Inverse of line_index_to_bitstring()."""
    if len(bits) != 3 or set(bits) - {"0", "1"}:
        raise ValueError(f"Expected a 3-bit string, got {bits!r}.")
    return int(bits, 2)
