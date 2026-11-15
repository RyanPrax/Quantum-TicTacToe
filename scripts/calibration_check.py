"""Check the in-job calibration against boards it was not fitted on.

    python scripts/calibration_check.py results/noise_bias_boards.json

The input is a noise_bias.py run with several boards per line. For each device
it replays calibrations from the run's own data: the 8 readout circuits plus,
as the known boards, one measured single win on each chosen line and the first
2 no-win boards. Every configuration is scored on the same held-out boards
(the single wins and no-win boards that no configuration uses):

  share error   rms error predicting each held-out single win's winner share
  missed        expected share of held-out wins missed (each board's measured
                share taken as its true share)
  flags         expected false flags per held-out no-win board
  asks for      the shots the profile predicts for at most 1% misses (capped)

once at 1024 shots and once at the shots the profile asks for. Configurations:
the previous calibration (one ε from the line-0 board, noise spread evenly)
and the per-bit calibration on lines 0, 1, 2 and 4 or on all 8 lines, each at
false-flag budgets of 0.1 and 0.2 checks per board. Then, for the last
configuration, each line's held-out winner share, its cutoff share and its
misses at 1024 and 8192 shots.
"""

import json
import sys

import numpy as np

from qttt.noise import NoiseProfile
from qttt.stats import NUM_LINES, binomial_below

SHOTS = 1024
MAX_SHOTS = 8192
CONFIGS = [
    # (name, calibration lines, per-bit, budget)
    ("one ε, line 0", (0,), False, 0.1),
    ("per bit, 0/1/2/4", (0, 1, 2, 4), True, 0.1),
    ("per bit, 0/1/2/4", (0, 1, 2, 4), True, 0.2),
    ("per bit, all 8", tuple(range(8)), True, 0.1),
    ("per bit, all 8", tuple(range(8)), True, 0.2),
]


def calibrate(run, singles, no_wins, per_bit):
    """A profile from the run's readout counts and the chosen calibration rows."""
    known = [(tuple(r["board"]), r["counts"]) for r in singles + (no_wins if per_bit else [])]
    return NoiseProfile.from_calibration(run["calibration_counts"][:NUM_LINES], known)


def score(profile, shots, budget, held_singles, held_no_wins, data_shots):
    """(expected share of held-out wins missed, expected false flags per held-out no-win board)."""
    cutoffs = profile.cutoffs(shots, budget)
    missed = np.mean([binomial_below(cutoffs[w], shots, r["counts"][w] / data_shots)
                      for r in held_singles for w in r["winning_lines"]])
    flags = np.mean([sum(1 - binomial_below(cutoffs[i], shots, r["counts"][i] / data_shots) for i in range(NUM_LINES))
                     for r in held_no_wins])
    return missed, flags


def check(run):
    singles = [r for r in run["rows"] if len(r["winning_lines"]) == 1]
    no_wins = [r for r in run["rows"] if not r["winning_lines"]]
    first = {line: next(r for r in singles if r["winning_lines"] == [line]) for line in range(NUM_LINES)}
    held_singles = [r for r in singles if r not in first.values()]
    held_no_wins = no_wins[2:]
    data_shots = run["shots"]
    print(f"\n{run['device']}  {run['mode']}: checked on {len(held_singles)} single wins and"
          f" {len(held_no_wins)} no-win boards, none used to calibrate")
    print(f"  {'calibration':18} {'budget':>6} {'share error':>11} | {'missed':>7} {'flags':>6} (1024 shots) |"
          f" {'asks for':>8} {'missed':>7} {'flags':>6}")
    for name, lines, per_bit, budget in CONFIGS:
        profile = calibrate(run, [first[line] for line in lines], no_wins[:2], per_bit)
        error = np.sqrt(np.mean([(profile.predicted_for({w})[w] - r["counts"][w] / data_shots) ** 2
                                 for r in held_singles for w in r["winning_lines"]]))
        missed, flags = score(profile, SHOTS, budget, held_singles, held_no_wins, data_shots)
        needed = profile.shots_needed(0.01, budget, MAX_SHOTS)
        missed_n, flags_n = score(profile, needed or MAX_SHOTS, budget, held_singles, held_no_wins, data_shots)
        asks = str(needed) if needed else f">{MAX_SHOTS}"
        print(f"  {name:18} {budget:>6} {error:>11.3f} | {missed:>7.1%} {flags:>6.3f} {'':12} |"
              f" {asks:>8} {missed_n:>7.1%} {flags_n:>6.3f}")
    print(f"  per line, {name}, budget {budget}: winner share (held-out mean), cutoff share, misses")
    print(f"    {'line':>4} {'winner':>7} {'cutoff@1024':>11} {'missed@1024':>11} {'cutoff@8192':>11}"
          f" {'missed@8192':>11}")
    low, high = profile.cutoffs(SHOTS, budget), profile.cutoffs(MAX_SHOTS, budget)
    for line in range(NUM_LINES):
        shares = [r["counts"][line] / data_shots for r in held_singles if r["winning_lines"] == [line]]
        miss_low = np.mean([binomial_below(low[line], SHOTS, s) for s in shares])
        miss_high = np.mean([binomial_below(high[line], MAX_SHOTS, s) for s in shares])
        print(f"    {line:>4} {np.mean(shares):>7.3f} {low[line] / SHOTS:>11.3f} {miss_low:>11.1%}"
              f" {high[line] / MAX_SHOTS:>11.3f} {miss_high:>11.1%}")


def main(argv=None):
    paths = (argv if argv is not None else sys.argv[1:]) or ["results/noise_bias_boards.json"]
    for path in paths:
        for run in json.load(open(path)):
            check(run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
