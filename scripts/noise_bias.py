"""Measure how noise spreads over the 8 lines, and how far the current noise model is off.

    python scripts/noise_bias.py                                   # 3 free-plan fakes, every mode
    python scripts/noise_bias.py --devices fake_kingston --modes scheduled --shots 4096
    python scripts/noise_bias.py --devices fake_kingston fake_fez --modes scheduled --boards-per-line 4 \
        --out results/noise_bias_boards.json

This runs, per device and timing mode, one job holding the calibration
circuits detect_winners uses (the 8 readout circuits and the known boards in
qttt.noise.CALIBRATION_BOARDS) and:

- one single-win board for each of the 8 lines,
- no-win boards,
- one double-win board for each of a few line pairs,

then prints each board's measured share on its winning lines (or on each line,
with no winner) next to what the profile fitted in the same job predicts. The
runs saved before the per-bit calibration (results/noise_bias.json,
noise_bias_boards.json, noise_bias_ancilla.json, noise_bias_score.json) used
one known board (winning line 0) and one ε for every line, as their commits'
code shows.

Timing modes for the simulator:
  plain      no idle decay: qubits only decay during gates
  scheduled  idle time padded with delays, so qubits also decay while waiting
             (the package default for fake backends)
  dd         as scheduled, with X-X dynamical decoupling pulses in the idle
             gaps (the closest the simulator gets to the hardware's XY4)

Raw counts go to --out as JSON. About 20 s per board per 2048 shots on a
laptop; every mode on 3 devices takes about an hour.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

from qiskit.circuit.library import XGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling

from qttt.backends import ANCILLA_FLAGS, DEFAULT_TRANSPILE_SEEDS, IBMBackend, get_backend
from qttt.board import EMPTY, O, X, boards_by_win_count, classical_winning_lines
from qttt.experiments import sample_boards
from qttt.noise import DEFAULT_KNOWN, calibration_circuits, profile_from_calibration
from qttt.results import provenance
from qttt.stats import NUM_LINES

DEVICES = ("fake_kingston", "fake_marrakesh", "fake_fez")
MODES = ("plain", "scheduled", "dd")
DOUBLE_PAIRS = ((0, 3), (1, 6), (2, 5), (4, 7))  # legal pairs covering every line index bit


class _DDBackend(IBMBackend):
    """Simulated IBM device that fills idle gaps with X-X decoupling pulses instead of bare delays."""

    def _sample(self, circuits, shots, on_submit):
        target = self.backend.target
        pm = PassManager([ALAPScheduleAnalysis(target=target),
                          PadDynamicalDecoupling(target=target, dd_sequence=[XGate(), XGate()])])
        return super()._sample(pm.run(list(circuits)), shots, on_submit)


def last_player(board):
    return X if board.count(X) > board.count(O) else O


def choose_boards(seed, boards_per_line=1):
    """(name, board, player, winning lines) for every case, drawn with a fixed seed.

    With one board per line (the first run, results/noise_bias.json) each line's board is drawn at random.
    With more, single wins come from sample_boards' balanced draw, so each line gets boards_per_line boards
    with alternating last mover and spread turns, which separates line effects from board effects, and
    the no-win boards cover every (last mover, turn).
    """
    rng = random.Random(seed)
    groups = boards_by_win_count()
    cases = []
    if boards_per_line == 1:
        for line in range(NUM_LINES):
            options = [b for b in groups[1] if classical_winning_lines(b, last_player(b)) == {line}]
            cases.append((f"single {line}", rng.choice(options), {line}))
        for i, board in enumerate(rng.sample(groups[0], 2)):
            cases.append((f"no win {i}", board, set()))
    else:
        singles = sample_boards(seed=seed, mix={"single_win": NUM_LINES * boards_per_line})
        for b in sorted(singles, key=lambda b: (min(b.lines), b.player, b.turn)):
            cases.append((f"single {min(b.lines)}", b.board, set(b.lines)))
        for i, b in enumerate(sample_boards(seed=seed, mix={"no_win": 5})):
            cases.append((f"no win {i}", b.board, set()))
    for pair in DOUBLE_PAIRS:
        options = [b for b in groups[2] if classical_winning_lines(b, last_player(b)) == set(pair)]
        cases.append((f"double {pair[0]},{pair[1]}", rng.choice(options), set(pair)))
    return [(name, board, last_player(board), lines) for name, board, lines in cases]


def make_backend(device, mode, seed, transpile_seeds, ancilla="auto"):
    if mode == "dd":
        plain = get_backend(device, seed=seed)
        return _DDBackend(plain.backend, seed=seed, transpile_seeds=transpile_seeds, schedule=False, ancilla=ancilla)
    backend = get_backend(device, seed=seed, transpile_seeds=transpile_seeds, ancilla=ancilla)
    backend.schedule = mode == "scheduled"
    return backend


def measure(device, mode, cases, shots, seed, transpile_seeds, ancilla="auto"):
    backend = make_backend(device, mode, seed, transpile_seeds, ancilla)
    extra, line_qubits = calibration_circuits(backend, "player", DEFAULT_KNOWN)
    boards = [backend.search_circuit("player", board, player) for _, board, player, _ in cases]
    start = time.time()
    out = backend.run(extra + boards, shots)
    profile = profile_from_calibration(backend, out.counts, line_qubits, DEFAULT_KNOWN)
    metrics = backend.metrics(backend.search_template("player"))
    rows = []
    for (name, board, player, lines), counts in zip(cases, out.counts[len(extra):], strict=True):
        predicted = profile.predicted_for(lines)
        rows.append({"case": name, "board": "".join(board), "player": player, "winning_lines": sorted(lines),
                     "turn": 9 - board.count(EMPTY), "set_qubits": board.count(player),  # board qubits in |1>
                     "counts": list(counts), "predicted": list(predicted)})
    return {"device": device, "mode": mode, "shots": shots, "seed": seed, "transpile_seeds": transpile_seeds,
            "ancilla": metrics.ancilla, "two_qubit_gates": metrics.two_qubit_gates, "log10_esp": metrics.log10_esp,
            "line_qubits": list(metrics.line_qubits), "epsilon": profile.epsilon, "depth": metrics.depth,
            "ancilla_setting": ancilla,
            "readout": [list(r) for r in profile.readout],
            "calibration_counts": [list(c) for c in out.counts[:len(extra)]],
            "seconds": round(time.time() - start), "provenance": provenance(), "rows": rows}


def report(result):
    shots = result["shots"]
    print(f"\n{result['device']}  {result['mode']}  ancilla {'yes' if result['ancilla'] else 'no'}"
          f" ({result['ancilla_setting']}), depth {result['depth']}"
          f"  ε fitted {result['epsilon']:.3f}  ({result['seconds']} s)")
    print(f"  {'case':12} {'measured on winners':>20} {'predicted':>10}   no-win: share per line")
    for row in result["rows"]:
        counts, lines = row["counts"], row["winning_lines"]
        if lines:
            measured = " / ".join(f"{counts[i] / shots:.3f}" for i in lines)
            predicted = " / ".join(f"{row['predicted'][i]:.3f}" for i in lines)
            print(f"  {row['case']:12} {measured:>20} {predicted:>10}")
        else:
            shares = " ".join(f"{c / shots:.3f}" for c in counts)
            print(f"  {row['case']:12} {'':>20} {'0.125 each':>10}   {shares}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--devices", nargs="+", default=list(DEVICES))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=4903, help="seed for choosing boards, compiling and simulating")
    parser.add_argument("--transpile-seeds", type=int, default=DEFAULT_TRANSPILE_SEEDS)
    parser.add_argument("--ancilla", nargs="+", choices=ANCILLA_FLAGS, default=["auto"],
                        help="ancilla settings to run, each separately (default auto)")
    parser.add_argument("--boards-per-line", type=int, default=1,
                        help="single-win boards per line (default 1; more separates line from board effects)")
    parser.add_argument("--out", default="results/noise_bias.json", help="where to save every count (JSON)")
    args = parser.parse_args(argv)

    provenance()  # cached now, so every result records the code as it was at start, not after later edits
    cases = choose_boards(args.seed, args.boards_per_line)
    results = []
    for device in args.devices:
        for mode in args.modes:
            for ancilla in args.ancilla:
                results.append(measure(device, mode, cases, args.shots, args.seed, args.transpile_seeds,
                                       ANCILLA_FLAGS[ancilla]))
                report(results[-1])
                out = Path(args.out)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(results, indent=1))  # saved after every run, so a stop loses nothing
    print(f"\nSaved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
