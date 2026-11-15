"""What saved calibration jobs say a device can do: miss rates and false flags, from measured shares only.

    python scripts/hardware_check.py ibm_marrakesh

Reads the calibration records saved for the device (no account or network
needed) and groups them by whether they ran with error suppression. For each
group it pools, per line, the share of shots on the winning line of the
calibration board that wins on that line, and the share each line gets on the
calibration boards without a winner. It does not use the fitted noise model,
whose bit factors are fitted to noise when little signal survives.

From those it sets each line's cutoff for the false-flag budget, as if the
pooled no-win shares were exact, and reports at several shot counts:

  single missed   chance a single win is not flagged: averaged over the 8 lines,
                  for the worst line, and weighted by how often each line is the
                  winning line in real games (the diagonals most often), also
                  for games won on a row or column only (lines 0-5)
  double missed   chance neither line of a double win is flagged (the winner is
                  missed), averaged over the legal pairs. No double win was
                  measured: each line of a double win is taken to keep
                  0.5 / (25/32) of a single win's signal above 1/8, as in the
                  noiseless circuit (0.375 vs 0.656)
  false flags     expected losing lines flagged per board without a winner,
                  and the chance such a board gets at least one

The same table follows with the cutoffs the detector itself would use, from
the profile fitted on the group's pooled calibrations (each line's cutoff sits
above the highest no-win share plus NO_WIN_MARGIN standard errors): what a
batch run with collect.py --calibration on these tags would do.

Each measured share comes from 1024 shots per job, so with few jobs the
winner's excess over 1/8 is uncertain; the "excess" column gives it in standard
errors.
"""

import sys

import numpy as np

from qttt.board import O, X, boards_by_win_count, classical_winning_lines
from qttt.noise import NoiseProfile, legal_double_wins
from qttt.results import RunStore
from qttt.stats import DEFAULT_BUDGET, NUM_LINES, binomial_below, binomial_cutoff

SHOTS = (1024, 4096, 16384, 65536)
DOUBLE_SCALE = (0.5 - 1 / 8) / (25 / 32 - 1 / 8)  # a double-win line's signal above 1/8, relative to a single win's


def game_weights():
    """Share of real games' single wins on each line (from every legal single-win board)."""
    counts = np.zeros(NUM_LINES)
    for board in boards_by_win_count()[1]:
        player = X if board.count(X) > board.count(O) else O
        (line,) = classical_winning_lines(board, player)
        counts[line] += 1
    return counts / counts.sum()


def pooled(calibrations):
    """Per line: (winner counts, winner shots), and no-win (counts, shots), pooled over the calibrations."""
    win, win_shots = np.zeros(NUM_LINES), np.zeros(NUM_LINES)
    none, none_shots = np.zeros(NUM_LINES), 0
    for cal in calibrations:
        for k in cal["known"]:
            lines = classical_winning_lines(k["board"], X) | classical_winning_lines(k["board"], O)
            counts = np.array(k["counts"], dtype=float)
            if len(lines) == 1:
                (w,) = lines
                win[w] += counts[w]
                win_shots[w] += counts.sum()
            elif not lines:
                none += counts
                none_shots += counts.sum()
    return win / win_shots, win_shots, none / none_shots, none_shots


def detector_cutoffs(calibrations):
    """The cutoffs function of the profile fitted on the pooled calibrations, as the detector would use."""
    readout = [[sum(c["readout_counts"][p][m] for c in calibrations) for m in range(NUM_LINES)]
               for p in range(NUM_LINES)]
    known = [(tuple(k["board"]), k["counts"]) for c in calibrations for k in c["known"]]
    profile = NoiseProfile.from_calibration(readout, known)
    return lambda n: profile.cutoffs(n, DEFAULT_BUDGET)


def report(name, calibrations):
    winner, win_shots, losing, _ = pooled(calibrations)
    eps = ", ".join(f"{c['profile']['epsilon']:.3f}" for c in calibrations)
    print(f"\n{name}: {len(calibrations)} job(s) (fitted ε {eps})")
    print(f"  {'line':>4} {'winner':>7} {'no-win':>7} {'excess':>7}")
    for i in range(NUM_LINES):
        se = np.sqrt(winner[i] * (1 - winner[i]) / win_shots[i])
        print(f"  {i:>4} {winner[i]:>7.3f} {losing[i]:>7.3f} {(winner[i] - 1 / 8) / se:>6.1f}σ")
    double = {i: 1 / 8 + DOUBLE_SCALE * (winner[i] - 1 / 8) for i in range(NUM_LINES)}
    weights = game_weights()
    straight = np.where(np.arange(NUM_LINES) < 6, weights, 0.0)
    straight /= straight.sum()
    rules = (("cutoffs from the mean no-win shares", lambda n: [binomial_cutoff(n, DEFAULT_BUDGET,
                                                                               min(max(q, 1e-9), 1 - 1e-9))
                                                                for q in losing]),
             ("the detector's cutoffs (collect.py --calibration)", detector_cutoffs(calibrations)))
    for rule, cutoffs_at in rules:
        print(f"  with {rule}:")
        table(cutoffs_at, winner, losing, double, weights, straight)


def table(cutoffs_at, winner, losing, double, weights, straight):
    print(f"  {'shots':>6} {'single missed (avg / worst line / games / no diagonals)':>56} {'double missed':>14}"
          f" {'false flags':>12} {'P(any)':>7}")
    for n in SHOTS:
        cutoffs = cutoffs_at(n)
        single = [binomial_below(cutoffs[i], n, winner[i]) for i in range(NUM_LINES)]
        both = [binomial_below(cutoffs[a], n, double[a]) * binomial_below(cutoffs[b], n, double[b])
                for a, b in legal_double_wins()]
        flags = [1 - binomial_below(cutoffs[i], n, losing[i]) for i in range(NUM_LINES)]
        any_flag = 1 - np.prod([1 - f for f in flags])
        print(f"  {n:>6} {np.mean(single):>22.1%} / {max(single):>7.1%} / {np.dot(weights, single):>6.1%}"
              f" / {np.dot(straight, single):>12.1%}"
              f" {np.mean(both):>14.1%}"
              f" {sum(flags):>12.3f} {any_flag:>7.1%}")


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    device = args[0] if args else "ibm_marrakesh"
    calibrations = [c for c in RunStore("results").load("calibration") if c["backend"]["name"] == device]
    if not calibrations:
        print(f"No calibration saved for {device}.")
        return 1
    for name, suppressed in (("error suppression on (DD + twirling)", True), ("error suppression off", False)):
        group = [c for c in calibrations if bool(c["backend"]["options"].get("sampler_options")) == suppressed]
        if group:
            report(name, group)
    report("all jobs pooled", calibrations)
    return 0


if __name__ == "__main__":
    sys.exit(main())
