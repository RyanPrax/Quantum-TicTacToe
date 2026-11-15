"""Grover oracles that phase-flip the line indices of winning lines."""

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit, QuantumRegister
from qiskit.synthesis import synth_mcx_1_clean_kg24

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


def player_oracle(
    qc: QuantumCircuit,
    board_reg: QuantumRegister,
    line_reg: QuantumRegister,
    ancilla: Qubit | None = None,
) -> None:
    """Oracle for the 12-qubit version: checks every line for the player who just moved.

    Expects board_reg loaded by encode_player_board() (9 qubits, qubit c is |1>
    when the player owns cell c). One multi-controlled Z per line on 6 qubits,
    the 3 line-index qubits and the line's 3 cell qubits, so it flips the sign
    of the line index directly; no phase qubit is needed. With ancilla, a qubit
    that starts in |0>, each MCZ is built with fewer gates, and the ancilla is
    returned to |0>.
    """
    for index, cells in enumerate(WIN_LINES):
        code = line_index_to_bitstring(index)

        # Flip line-index qubits whose bit is 0, so the gate matches only this index.
        line_controls = []
        temp_line_flips = []
        for q_index, bit in enumerate(reversed(code)):
            if bit == "0":
                qc.x(line_reg[q_index])
                temp_line_flips.append(line_reg[q_index])
            line_controls.append(line_reg[q_index])

        # An owned cell is already |1>, so the cell qubits take part as they are.
        mcz(qc, line_controls + [board_reg[cell] for cell in cells], ancilla)

        for qubit in reversed(temp_line_flips):
            qc.x(qubit)


def mcz(qc: QuantumCircuit, qubits: list[Qubit], ancilla: Qubit | None = None) -> None:
    """Flip the sign of the state where every qubit is |1>: an MCX on the last qubit between two H gates.

    With ancilla (in |0>), the MCX uses Khattar-Gidney's one-clean-ancilla synthesis.
    """
    *controls, target = qubits
    qc.h(target)
    if ancilla is None:
        qc.mcx(controls, target)
    else:
        qc.compose(synth_mcx_1_clean_kg24(len(controls)), [*controls, target, ancilla], inplace=True)
    qc.h(target)
