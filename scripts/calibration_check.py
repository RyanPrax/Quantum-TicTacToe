"""Check the in-job calibration against boards it was not fitted on.

    python scripts/calibration_check.py results/noise_bias_boards.json

The input is a noise_bias.py run with several boards per line. For each device
it replays a calibration from the run's own data: the 8 readout circuits plus,
as the known boards, one measured single win on each of lines 0, 1, 2 and 4
and the first 2 no-win boards (what detect_winners measures in every job). It
compares two profiles fitted that way:

  one ε     fitted on the line-0 board only, noise spread evenly (the previous calibration)
  per bit   ε and a signal factor per line-index bit, plus cutoffs raised to the
            highest shares measured on the no-win boards (the current calibration)

on every other board in the run: the error predicting each held-out single
win's winner share, the expected share of held-out wins missed at 1024 shots
(each board's measured share taken as its true share), and the expected false
flags per held-out no-win board under each profile's cutoffs.

It also compares ways of turning the no-win boards into cutoffs: the highest
share per line or the mean over the boards, plus 0, 1 or 2 standard errors.
"""

import json
import sys
from dataclasses import replace

import numpy as np

from qttt import noise
from qttt.noise import NoiseProfile
from qttt.stats import NUM_LINES, binomial_below

SHOTS = 1024
BUDGET = 0.1


def board(row):
    return tuple(row["board"])


def calibrate(run, rows, per_bit):
    """A profile from the run's readout counts and the chosen calibration rows."""
    known = [(board(r), r["counts"]) for r in rows if per_bit or r["winning_lines"] == [0]]
    return NoiseProfile.from_calibration(run["calibration_counts"][:NUM_LINES], known)


def check(run):
    singles = [r for r in run["rows"] if len(r["winning_lines"]) == 1]
    no_wins = [r for r in run["rows"] if not r["winning_lines"]]
    chosen = [next(r for r in singles if r["winning_lines"] == [line]) for line in (0, 1, 2, 4)] + no_wins[:2]
    held_singles = [r for r in singles if r not in chosen]
    held_no_wins = [r for r in no_wins if r not in chosen]
    shots = run["shots"]
    print(f"\n{run['device']}  {run['mode']}: calibrated on {len(chosen)} boards, checked on {len(held_singles)}"
          f" single wins and {len(held_no_wins)} no-win boards")
    print(f"  {'profile':9} {'ε':>6}  {'bit factors':18} {'share error':>11} {'wins missed':>11}"
          f" {'false flags/board':>17}")
    for name, per_bit in (("one ε", False), ("per bit", True)):
        profile = calibrate(run, chosen, per_bit)
        cutoffs = profile.cutoffs(SHOTS, BUDGET)
        errors, missed = [], []
        for r in held_singles:
            w = r["winning_lines"][0]
            share = r["counts"][w] / shots
            errors.append(profile.predicted_for({w})[w] - share)
            missed.append(binomial_below(cutoffs[w], SHOTS, share))
        flags = [sum(1 - binomial_below(cutoffs[i], SHOTS, r["counts"][i] / shots) for i in range(NUM_LINES))
                 for r in held_no_wins]
        factors = ", ".join(f"{f:.3f}" for f in profile.bit_factors)
        print(f"  {name:9} {profile.epsilon:>6.3f}  {factors:18} {np.sqrt(np.mean(np.square(errors))):>11.3f}"
              f" {np.mean(missed):>11.2%} {np.mean(flags):>17.3f}")
    print(f"  (false-flag budget: {BUDGET} per board)")
    base = calibrate(run, chosen, per_bit=True)
    calibration_no_wins = [r for r in chosen if not r["winning_lines"]]
    print("  no-win cutoff rule        wins missed  false flags/board")
    for rule in ("max", "mean"):
        shares = [[r["counts"][i] / shots for r in calibration_no_wins] for i in range(NUM_LINES)]
        agg = tuple((max if rule == "max" else np.mean)(v) for v in shares)
        n = shots if rule == "max" else shots * len(calibration_no_wins)
        for margin in (2.0, 1.0, 0.0):
            noise.NO_WIN_MARGIN = margin
            profile = replace(base, no_win_shares=tuple(float(a) for a in agg), no_win_shots=n)
            cutoffs = profile.cutoffs(SHOTS, BUDGET)
            missed = np.mean([binomial_below(cutoffs[r["winning_lines"][0]], SHOTS,
                                             r["counts"][r["winning_lines"][0]] / shots) for r in held_singles])
            flags = np.mean([sum(1 - binomial_below(cutoffs[i], SHOTS, r["counts"][i] / shots)
                                 for i in range(NUM_LINES)) for r in held_no_wins])
            print(f"    {rule} + {margin:.0f} SE{'':14} {missed:>11.2%} {flags:>17.3f}")
    noise.NO_WIN_MARGIN = 2.0
    print("  current calibration at more shots per board (shares are per shot, so the run's data still applies):")
    print(f"    {'shots':>5} {'wins missed':>11} {'false flags/board':>17}   profile asks for")
    for n in (1024, 2048, 4096):
        cutoffs = base.cutoffs(n, BUDGET)
        missed = np.mean([binomial_below(cutoffs[r["winning_lines"][0]], n, r["counts"][r["winning_lines"][0]] / shots)
                          for r in held_singles])
        flags = np.mean([sum(1 - binomial_below(cutoffs[i], n, r["counts"][i] / shots) for i in range(NUM_LINES))
                         for r in held_no_wins])
        print(f"    {n:>5} {missed:>11.2%} {flags:>17.3f}")
    print(f"    shots for at most 1% misses predicted by the profile: {base.shots_needed(0.01, BUDGET)}")
    pooled = calibrate(run, singles + [r for r in chosen if not r["winning_lines"]], per_bit=True)
    print(f"    the same with ε and factors fitted on all {len(singles)} single wins (factors"
          f" {', '.join(f'{f:.3f}' for f in pooled.bit_factors)}): {pooled.shots_needed(0.01, BUDGET)}")


def main(argv=None):
    paths = (argv if argv is not None else sys.argv[1:]) or ["results/noise_bias_boards.json"]
    for path in paths:
        for run in json.load(open(path)):
            check(run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
