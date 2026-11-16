"""Analyze a noise_bias.py run: is the winner's signal set by its line, or by the board?

    python scripts/noise_analysis.py results/noise_bias_boards.json

For every device and mode in the file (best with --boards-per-line 4 or more):

1. Spread of the winning line's share of shots on single-win boards: between
   lines (the spread of the 8 per-line means), within a line (between boards
   that win on the same line), and what shot noise alone would give.
2. How well simple models predict a board's winner share, each scored by
   leave-one-out (fit on the other boards, predict this one), so a model with
   more numbers gains nothing from fitting noise:
     current   the noise profile fitted in the job on one known board (line 0)
     constant  one share for every board (one ε fitted on many boards)
     |1> count share depending linearly on how many board qubits start in |1>
     per bit   the shipped model (qttt.noise.fit_line_signal): signal (1 − ε) × f0^bit0 × f1^bit1 ×
               f2^bit2 above the 1/8 floor, through the run's readout matrix (4 numbers)
     per line  one share per winning line
3. Losing lines: the highest share any losing line reached, on no-win boards
   and on single-win boards; the false-flag cutoffs must sit above these.
"""

import json
import sys
from statistics import mean, stdev

import numpy as np

from qttt.noise import NoiseProfile


def winner_rows(run):
    shots = run["shots"]
    rows = []
    for r in run["rows"]:
        if len(r["winning_lines"]) == 1:
            line = r["winning_lines"][0]
            rows.append({"line": line, "ones": r["set_qubits"], "turn": r["turn"], "player": r["player"],
                         "share": r["counts"][line] / shots, "current": r["predicted"][line],
                         "board": tuple(r["board"]), "counts": r["counts"],
                         "readout_counts": run["calibration_counts"][:8]})
    return rows


def loo_rms(rows, predict):
    """Root-mean-square error of predict(training rows, row) with each row left out in turn."""
    errors = [predict(rows[:i] + rows[i + 1:], row) - row["share"] for i, row in enumerate(rows)]
    return float(np.sqrt(np.mean(np.square(errors))))


def constant(train, row):
    return mean(r["share"] for r in train)


def by_ones(train, row):
    slope, intercept = np.polyfit([r["ones"] for r in train], [r["share"] for r in train], 1)
    return slope * row["ones"] + intercept


def _fitted(rows):
    """The detector's own profile (readout matrix, ε and bit factors) fitted on these single wins."""
    return NoiseProfile.from_calibration(rows[0]["readout_counts"], [(r["board"], r["counts"]) for r in rows])


def by_bit(train, row):
    return _fitted(train).predicted_for({row["line"]})[row["line"]]


def bit_factors(rows):
    """(1 − ε, factor kept per 1 in bit 0, 1, 2) fitted on every row, with the detector's fit."""
    profile = _fitted(rows)
    return [1 - profile.epsilon, *profile.bit_factors]


def by_line(train, row):
    same = [r["share"] for r in train if r["line"] == row["line"]]
    return mean(same) if same else constant(train, row)


def analyze(run):
    rows = winner_rows(run)
    shots = run["shots"]
    lines = sorted({r["line"] for r in rows})
    per_line = {line: [r["share"] for r in rows if r["line"] == line] for line in lines}
    overall = mean(r["share"] for r in rows)
    print(f"\n{run['device']}  {run['mode']}  ancilla {'yes' if run['ancilla'] else 'no'}"
          f"  {len(rows)} single wins, {shots} shots each")
    print("  winner share per line (mean of its boards; each board):")
    for line in lines:
        print(f"    line {line}: {mean(per_line[line]):.3f}   " + " ".join(f"{s:.3f}" for s in per_line[line]))
    between = stdev([mean(v) for v in per_line.values()])
    within = mean(stdev(v) for v in per_line.values() if len(v) > 1)
    noise = (overall * (1 - overall) / shots) ** 0.5
    print(f"  spread: between lines {between:.3f}, between boards of one line {within:.3f},"
          f" shot noise alone {noise:.3f}")
    ones = np.corrcoef([r["ones"] for r in rows], [r["share"] for r in rows])[0, 1]
    print(f"  correlation of share with |1> board qubits: {ones:+.2f}")
    current = float(np.sqrt(np.mean([(r["current"] - r["share"]) ** 2 for r in rows])))
    print(f"  prediction error (rms): current {current:.3f}, constant {loo_rms(rows, constant):.3f},"
          f" |1> count {loo_rms(rows, by_ones):.3f}, per bit {loo_rms(rows, by_bit):.3f},"
          f" per line {loo_rms(rows, by_line):.3f}")
    base, *factors = bit_factors(rows)
    print(f"  per-bit fit: signal kept 1 − ε = {base:.3f}, factor kept per 1 in bit 0, 1, 2:"
          f" {', '.join(f'{f:.3f}' for f in factors)} (line qubits {run['line_qubits']})")
    losing_single = max(c / shots for r in run["rows"] if len(r["winning_lines"]) == 1
                        for i, c in enumerate(r["counts"]) if i not in r["winning_lines"])
    no_win = [r for r in run["rows"] if not r["winning_lines"]]
    losing_none = max(c / shots for r in no_win for c in r["counts"]) if no_win else float("nan")
    print(f"  highest losing-line share: no-win boards {losing_none:.3f}, single-win boards {losing_single:.3f}")


def main(argv=None):
    paths = (argv if argv is not None else sys.argv[1:]) or ["results/noise_bias_boards.json"]
    for path in paths:
        with open(path, encoding="utf-8") as f:
            runs = json.load(f)
        for run in runs:
            analyze(run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
