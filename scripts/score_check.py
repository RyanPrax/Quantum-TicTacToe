"""Check which template score ranks compiled circuits the way measured signal does.

    python scripts/score_check.py results/noise_bias_ancilla.json

The input is a noise_bias.py run with --ancilla on off (made with the template
choice that scores gate ESP only). For each device this rebuilds both measured
templates the same way, checks they match what was run (two-qubit gates, depth,
line qubits), and scores each with:

  esp           gate ESP only (what chose the measured templates)
  esp+idle      plus idle survival: exp(-t / T2) for every idle stretch after
                a qubit's first gate, T2 capped at 2 T1, on every qubit
  esp+idle_line plus the same, counted only on the 3 line qubits and the
                ancilla (the qubits held in superposition the whole time)
  ... +readout  plus the 3 line qubits' readout success

and prints which version each score picks next to which one kept more signal.
"""

import json
import math
import sys
from statistics import mean

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDelay

from qttt.backends import IBMBackend, get_backend, line_qubits, two_qubit_count
from qttt.board import EMPTY, X
from qttt.grover import build_search_circuit


class _EspOnly(IBMBackend):
    """The template choice the measured run used: gate ESP, then fewer two-qubit gates."""

    def score(self, tqc):
        esp = self.gate_log10_esp(tqc)
        return (esp if esp is not None else 0.0, -two_qubit_count(tqc))


def idle_log10(backend, tqc, only=None):
    """log10 idle survival, on every qubit or only on the physical qubits in only."""
    target = backend.backend.target
    scheduled = PassManager([ALAPScheduleAnalysis(target=target), PadDelay(target=target)]).run(tqc)
    started, total = set(), 0.0
    for inst in scheduled.data:
        qubits = [scheduled.find_bit(q).index for q in inst.qubits]
        if inst.operation.name != "delay":
            started.update(qubits)
            continue
        q = qubits[0]
        p = target.qubit_properties[q]
        if q not in started or (only is not None and q not in only) or p.t1 is None or p.t2 is None:
            continue
        seconds = inst.operation.duration * (target.dt if inst.operation.unit == "dt" else 1)
        total -= seconds / min(p.t2, 2 * p.t1) / math.log(10)
    return total


def exposed_qubits(tqc, ancilla):
    """Physical qubits holding the line register (at the start and when measured) and the ancilla."""
    layout = tqc.layout.initial_index_layout(filter_ancillas=True)
    exposed = set(layout[9:12]) | set(line_qubits(tqc))
    if ancilla:
        exposed.add(layout[12])
    return exposed


def scores(backend, tqc, ancilla):
    gate = backend.gate_log10_esp(tqc)
    idle = idle_log10(backend, tqc)
    idle_line = idle_log10(backend, tqc, exposed_qubits(tqc, ancilla))
    readout = backend.readout_log10_success(tqc) if hasattr(backend, "readout_log10_success") else 0.0
    readout = readout or 0.0
    return {"esp": gate, "esp+idle": gate + idle, "esp+idle_line": gate + idle_line,
            "esp+readout": gate + readout, "esp+idle+readout": gate + idle + readout,
            "esp+idle_line+readout": gate + idle_line + readout}


def main(argv=None):
    path = (argv if argv is not None else sys.argv[1:])[0]
    runs = json.load(open(path))
    by_device = {}
    for run in runs:
        shots = run["shots"]
        signal = mean(r["counts"][r["winning_lines"][0]] / shots for r in run["rows"] if len(r["winning_lines"]) == 1)
        by_device.setdefault(run["device"], {})[run["ancilla_setting"]] = (run, signal)
    names = None
    for device, settings in by_device.items():
        row = {}
        for ancilla, (run, signal) in settings.items():
            plain = get_backend(device, seed=run["seed"])
            backend = _EspOnly(plain.backend, seed=run["seed"], transpile_seeds=run["transpile_seeds"], ancilla=ancilla)
            tqc = backend.transpile(build_search_circuit("player", [EMPTY] * 9, X, ancilla=ancilla))
            m = backend.metrics(tqc)
            same = (m.two_qubit_gates, m.depth, list(m.line_qubits)) == (run["two_qubit_gates"], run["depth"],
                                                                         run["line_qubits"])
            row[ancilla] = (signal, scores(backend, tqc, ancilla), same)
        names = names or list(row[True][1])
        better = max(row, key=lambda a: row[a][0])
        print(f"\n{device}: measured signal with ancilla {row[True][0]:.3f}, without {row[False][0]:.3f}"
              f" -> {'ancilla' if better else 'no ancilla'} is better"
              f" (templates reproduced: {row[True][2] and row[False][2]})")
        for name in names:
            pick = max(row, key=lambda a: row[a][1][name])
            mark = "right" if pick == better else "WRONG"
            print(f"  {name:22} with {10 ** row[True][1][name]:.2e}  without {10 ** row[False][1][name]:.2e}"
                  f"  picks {'ancilla' if pick else 'no ancilla':10}  {mark}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
