"""What saved calibration jobs say a device can do: miss rates and false flags, from measured shares only.

    python scripts/hardware_check.py ibm_marrakesh

Reads the calibration records saved for the device (no account or network
needed) and groups them by whether they ran with error suppression; groups are
never pooled with each other, since suppression changes the noise. For each
group it pools, per line, the share of shots on the winning line of the
calibration board that wins on that line, and the share each line gets on the
calibration boards without a winner. It does not use the fitted noise model.

From those it sets each line's cutoff for the false-flag budget, as if the
pooled no-win shares were exact, and reports at several shot counts:

  single missed   chance a single win is not flagged: averaged over the 8 lines,
                  for the worst line, and weighted by how often each line is the
                  winning line over every possible game (each complete game of
                  tic-tac-toe counted once; the diagonals most often), also
                  for games won on a row or column only (lines 0-5)
  double missed   chance neither line of a double win is flagged (the winner is
                  missed), averaged over the legal pairs, with both lines
                  sharing the same shots (binomial_both_below). Each line of a
                  double win is taken to keep 0.5 / (25/32) of a single win's
                  signal above 1/8, as in the noiseless circuit (0.375 vs
                  0.656); calibrations saved before the double-win calibration
                  boards measured none
  false flags     expected losing lines flagged per board without a winner,
                  and the chance such a board gets at least one

The same table follows with the cutoffs the detector itself would use, from
the profile fitted on the group's pooled calibrations (each line's cutoff sits
above the pooled mean share it gets as a loser plus NO_WIN_MARGIN standard
errors, see qttt.noise.losing_shares): what a batch run with collect.py
--calibration on these tags would do; and once more as if each of the jobs
had calibrated at the table's shot count, as a batch that calibrates in its
own jobs (the default) does.

Each measured share comes from 1024 shots per job, so with few jobs the
winner's excess over 1/8 is uncertain; the "excess" column gives it in standard
errors.
"""

import sys
from functools import cache

import numpy as np

from qttt.board import EMPTY, O, X, classical_winning_lines
from qttt.noise import NoiseProfile, legal_double_wins
from qttt.results import RunStore
from qttt.stats import DEFAULT_BUDGET, NUM_LINES, binomial_below, binomial_both_below, binomial_cutoff

SHOTS = (1024, 4096, 16384, 65536)
DOUBLE_SCALE = (0.5 - 1 / 8) / (25 / 32 - 1 / 8)  # a double-win line's signal above 1/8, relative to a single win's


def game_weights():
    """Share of single-line wins on each line, over every complete game of tic-tac-toe (each counted once)."""
    @cache
    def games(board: str) -> tuple[float, ...]:
        """Single-line wins per line over every way to play on from this board (X to move on an even count)."""
        player = X if board.count(EMPTY) % 2 == 1 else O
        total = np.zeros(NUM_LINES)
        for cell in (i for i, mark in enumerate(board) if mark == EMPTY):
            after = board[:cell] + player + board[cell + 1:]
            lines = classical_winning_lines(after, player)
            if len(lines) == 1:
                total[min(lines)] += 1
            elif not lines and EMPTY in after:
                total += games(after)
        return tuple(total)

    counts = np.array(games(EMPTY * 9))
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
    missing = [i for i in range(NUM_LINES) if not win_shots[i]]
    if missing or not none_shots:
        raise ValueError(f"the calibrations have no single-win board for lines {missing}" if missing
                         else "the calibrations have no board without a winner")
    return win / win_shots, win_shots, none / none_shots, none_shots


def detector_cutoffs(calibrations, in_job=False):
    """The cutoffs function of the profile fitted on the pooled calibrations, as the detector would use.

    With in_job, as if each job calibrated at its own shot count (the default
    for detect_winners): the measured losing shares' standard errors shrink
    with the shots (NoiseProfile.with_calibration_shots).
    """
    readout = [[sum(c["readout_counts"][p][m] for c in calibrations) for m in range(NUM_LINES)]
               for p in range(NUM_LINES)]
    known = [(tuple(k["board"]), k["counts"]) for c in calibrations for k in c["known"]]
    profile = NoiseProfile.from_calibration(readout, known)
    jobs = len(calibrations)
    if in_job:  # n shots per circuit in each of the pooled jobs
        return lambda n: profile.with_calibration_shots(n * jobs).cutoffs(n, DEFAULT_BUDGET)
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
             ("the detector's cutoffs (collect.py --calibration)", detector_cutoffs(calibrations)),
             ("the detector's cutoffs, calibrating in the job at the same shots (the default)",
              detector_cutoffs(calibrations, in_job=True)))
    for rule, cutoffs_at in rules:
        print(f"  with {rule}:")
        table(cutoffs_at, winner, losing, double, weights, straight)


def table(cutoffs_at, winner, losing, double, weights, straight):
    print(f"  {'shots':>6} {'single missed (avg / worst line / games / no diagonals)':>56} {'double missed':>14}"
          f" {'false flags':>12} {'P(any)':>7}")
    for n in SHOTS:
        cutoffs = cutoffs_at(n)
        single = [binomial_below(cutoffs[i], n, winner[i]) for i in range(NUM_LINES)]
        both = [binomial_both_below(n, cutoffs[a], double[a], cutoffs[b], double[b]) for a, b in legal_double_wins()]
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
