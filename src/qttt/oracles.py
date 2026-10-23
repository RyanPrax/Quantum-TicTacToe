"""Grover oracles that phase-flip the line indices of winning lines."""

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit, QuantumRegister

from qttt.board import WIN_LINES
from qttt.encoding import cell_bits, line_index_to_bitstring


def full_oracle(
    qc: QuantumCircuit,
    board_reg: QuantumRegister,
    line_reg: QuantumRegister,
    phase_qubit: Qubit,
) -> None:
    """Oracle for the 22-qubit version: checks every line for both X and O.

    Expects board_reg loaded by encode_full_board() (18 qubits) and
    phase_qubit prepared in |->, so each MCX acts as a phase flip on the
    line index (phase kickback).
    """
    for index, cells in enumerate(WIN_LINES):
        code = line_index_to_bitstring(index)

        # Flip line-index qubits whose bit is 0, so the controls match only this index.
        line_controls = []
        temp_line_flips = []
        for q_index, bit in enumerate(reversed(code)):
            if bit == "0":
                qc.x(line_reg[q_index])
                temp_line_flips.append(line_reg[q_index])
            line_controls.append(line_reg[q_index])

        # X win check: X is |10>, so flip q0 to make both qubits 1.
        x_controls = line_controls.copy()
        temp_flips_x = []
        for cell in cells:
            q0, q1 = cell_bits(cell)
            qc.x(board_reg[q0])
            temp_flips_x.append(board_reg[q0])
            x_controls.append(board_reg[q0])
            x_controls.append(board_reg[q1])

        qc.mcx(x_controls, phase_qubit)

        for qubit in reversed(temp_flips_x):
            qc.x(qubit)

        # O win check: O is |01>, so flip q1 to make both qubits 1.
        o_controls = line_controls.copy()
        temp_flips_o = []
        for cell in cells:
            q0, q1 = cell_bits(cell)
            qc.x(board_reg[q1])
            temp_flips_o.append(board_reg[q1])
            o_controls.append(board_reg[q0])
            o_controls.append(board_reg[q1])

        qc.mcx(o_controls, phase_qubit)

        for qubit in reversed(temp_flips_o):
            qc.x(qubit)

        for qubit in reversed(temp_line_flips):
            qc.x(qubit)
