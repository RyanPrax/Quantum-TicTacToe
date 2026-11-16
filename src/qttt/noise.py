"""Noise profiles: per-line cutoffs and predicted miss rates for a backend.

A profile models what one round measures as

    measured = A · p,    p_w = s_w · ideal_w + (1 − s_w) / 8  for each winning line w,

with the losing lines sharing the rest evenly. ideal is the noiseless line
distribution for the board, A[measured][prepared] is the 8×8 readout matrix of
the 3 line qubits, and s_w is the share of the winner's signal that survives:

    s_w = (1 − ε) · f₀^bit₀(w) · f₁^bit₁(w) · f₂^bit₂(w)

ε is the share of shots gate noise scatters, and each bit of the line index
that is 1 keeps a factor f_b of the signal. The winning line sets the signal,
not the rest of the board (scripts/noise_analysis.py on
results/noise_bias_boards.json), and these 4 numbers predict it at the
shot-noise floor. With every f = 1 this is measured = A · [(1 − ε) · ideal + ε / 8].
Each line of a double win keeps double_factor times its single-win signal,
measured on the double-win calibration boards.

From that each line gets its own cutoff, high enough that a losing line rarely
reaches it: above the model's worst case, and above the pooled mean share the
line was measured at on calibration boards where it loses (as a no-win board's
share), plus NO_WIN_MARGIN standard errors of that mean and of the spread
between boards. The profile then predicts how often wins are missed before any
hardware time is spent.

Profiles come from no noise (from_ideal), from a backend's published
calibration data (from_backend), or from calibration runs on the backend
itself (calibrate), which works on any provider.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property, lru_cache
from math import asin, sin, sqrt

import numpy as np

from qttt.backends import Backend, template_fingerprint
from qttt.board import EMPTY, O, X, boards_by_win_count, classical_winning_lines
from qttt.stats import DEFAULT_BUDGET, NUM_LINES, binomial_below, binomial_both_below, binomial_cutoff

_MIN_Q0 = 1e-12  # keeps binomial_cutoff's q0 strictly inside (0, 1)
_MIN_FACTOR = 1e-3  # a bit factor fitted at 0 still gets a finite log during the fit
_MAX_FACTOR = 2.0  # a bit factor above 1 is possible (noise), but not without bound
SIGNAL_SIGMAS = 3.0  # below this many standard errors of signal, fit one ε instead of the bit factors
FACTOR_SIGMAS = 2.0  # a bit factor closer to 1 than this many standard errors is set to 1

# Legal boards with known answers, measured in every calibrating job: a single
# win for X on each of the 8 lines (X just moved, turn 5), so each bit of the
# line index is 1 on four of them and ε and the 3 bit factors are fitted from
# all eight (scripts/calibration_check.py: twice as precise as one board per
# bit); two boards without a winner (X just moved on turn 5, O on turn 6) that
# measure how high losing lines get; and two double wins for X (lines 0 and 7,
# which differ in every bit, and lines 3 and 6) that measure how much signal
# each line of a double win keeps.
CALIBRATION_BOARDS = (
    ((X, X, X,
      O, O, EMPTY,
      EMPTY, EMPTY, EMPTY), X),
    ((O, O, EMPTY,
      X, X, X,
      EMPTY, EMPTY, EMPTY), X),
    ((O, O, EMPTY,
      EMPTY, EMPTY, EMPTY,
      X, X, X), X),
    ((X, O, O,
      X, EMPTY, EMPTY,
      X, EMPTY, EMPTY), X),
    ((O, X, O,
      EMPTY, X, EMPTY,
      EMPTY, X, EMPTY), X),
    ((O, O, X,
      EMPTY, EMPTY, X,
      EMPTY, EMPTY, X), X),
    ((X, O, O,
      EMPTY, X, EMPTY,
      EMPTY, EMPTY, X), X),
    ((O, O, X,
      EMPTY, X, EMPTY,
      X, EMPTY, EMPTY), X),
    ((X, O, EMPTY,
      EMPTY, O, X,
      EMPTY, X, EMPTY), X),
    ((X, O, X,
      O, X, EMPTY,
      EMPTY, EMPTY, O), O),
    ((X, X, X,
      O, X, O,
      X, O, O), X),
    ((X, O, O,
      X, X, O,
      X, O, X), X),
)
NO_WIN_MARGIN = 2.0  # standard errors added to a measured no-win share before it sets a cutoff


def ideal_line_probabilities(winners: Sequence[int] | set[int]) -> tuple[float, ...]:
    """Noiseless line distribution after one Grover iteration with these winning lines.

    With M of 8 lines marked and sin θ = √(M/8), the winners share sin²(3θ)
    and the other lines share the rest: 25/32 on a single winner, 1/2 on each
    of two, and 1/8 everywhere with none.
    """
    winners = set(winners)
    m = len(winners)
    if m == 0:
        return (1 / NUM_LINES,) * NUM_LINES
    p_win = sin(3 * asin(sqrt(m / NUM_LINES))) ** 2
    return tuple(p_win / m if i in winners else (1 - p_win) / (NUM_LINES - m) for i in range(NUM_LINES))


@lru_cache(maxsize=1)
def legal_double_wins() -> tuple[tuple[int, int], ...]:
    """Every pair of lines that can be completed together in a real game."""
    pairs = set()
    for board in boards_by_win_count()[2]:
        pairs.add(tuple(sorted(classical_winning_lines(board, X))))
    return tuple(sorted(pairs))


@dataclass(frozen=True)
class MissRates:
    """Predicted outcome of one round at a given shot count and budget."""
    shots: int
    budget: float
    cutoffs: tuple[int, ...]
    single: tuple[float, ...]  # P(miss) for a single win on each line
    double_both: dict[tuple[int, int], float]  # P(neither line flagged): the winner is missed
    double_any: dict[tuple[int, int], float]  # P(not both flagged): secondary
    false_flags_no_win: float  # expected rejected checks on a board with no winner, at the measured no-win shares
    undetectable: tuple[int, ...]  # lines whose winner looks no more likely than a loser

    @property
    def worst(self) -> float:
        """The largest chance of missing a win, over every single and double win."""
        return max([*self.single, *self.double_both.values()])


@dataclass(frozen=True)
class NoiseProfile:
    """Gate noise ε, the signal factor per line-index bit, and readout matrix A for one backend's line qubits."""
    epsilon: float
    readout: tuple[tuple[float, ...], ...]  # A[measured][prepared]
    source: str
    line_qubits: tuple[int, ...] | None = None  # physical qubits the readout matrix belongs to
    log10_esp: float | None = field(default=None, compare=False)
    # The saved calibration record this profile was fitted from, so every run that reuses it links back to it.
    calibration_id: str | None = field(default=None, compare=False)
    bit_factors: tuple[float, float, float] = (1.0, 1.0, 1.0)  # signal kept per 1 in bit 0, 1, 2 of the line
    # Each line's share as a loser, pooled over the calibration boards (see losing_shares), and those boards'
    # total shots. Without no_win_errors (records saved before them), the standard error is √(q(1 − q) / shots).
    no_win_shares: tuple[float, ...] | None = None
    no_win_shots: int = 0
    no_win_errors: tuple[float, ...] | None = None  # standard error of each pooled share
    no_win_spread: tuple[float, ...] | None = None  # spread of the true share between boards, beyond shot noise
    double_factor: float = 1.0  # signal each line of a double win keeps, relative to a single win on it
    # Shots per calibration circuit, so a profile can say what a calibration with more shots would give.
    calibration_shots: int = field(default=0, compare=False)
    # Standard errors of (ε, f₀, f₁, f₂) from the fit; 0 for a number that was not fitted.
    signal_errors: tuple[float, ...] | None = field(default=None, compare=False)
    # The compiled search circuit the profile belongs to (backends.template_fingerprint), if known.
    template: dict | None = field(default=None, compare=False)

    def __post_init__(self):
        if not 0 <= self.epsilon <= 1:
            raise ValueError(f"epsilon must be in [0, 1], got {self.epsilon}")
        if len(self.readout) != NUM_LINES or any(len(row) != NUM_LINES for row in self.readout):
            raise ValueError("readout must be an 8x8 matrix")
        if len(self.bit_factors) != 3 or not all(f > 0 for f in self.bit_factors):
            raise ValueError(f"bit_factors must be 3 positive numbers, got {self.bit_factors}")
        if self.no_win_shares is not None and (len(self.no_win_shares) != NUM_LINES or self.no_win_shots < 1):
            raise ValueError("no_win_shares needs one share per line and no_win_shots of at least 1")
        for name in ("no_win_errors", "no_win_spread"):
            values = getattr(self, name)
            if values is not None and (self.no_win_shares is None or len(values) != NUM_LINES):
                raise ValueError(f"{name} needs no_win_shares and one value per line")
        if self.double_factor < 0:
            raise ValueError(f"double_factor must be at least 0, got {self.double_factor}")

    # --- construction ---

    @classmethod
    def from_ideal(cls) -> "NoiseProfile":
        """No noise: every losing line gets at most 1/8, the same cutoff as before."""
        identity = tuple(tuple(1.0 if r == c else 0.0 for c in range(NUM_LINES)) for r in range(NUM_LINES))
        return cls(epsilon=0.0, readout=identity, source="ideal")

    @classmethod
    def from_backend(cls, backend: Backend, variant: str = "player") -> "NoiseProfile":
        """From published calibration data: ε = 1 − ESP of the search circuit, A from readout errors.

        Without published gate errors ε is 0; without readout errors A is the
        identity. Use calibrate() for providers that publish neither.
        """
        template = backend.search_template(variant)
        metrics = backend.metrics(template)
        epsilon = 0.0 if metrics.log10_esp is None else 1 - 10 ** metrics.log10_esp
        errors = backend.readout_errors(metrics.line_qubits)
        readout = readout_matrix_from_errors(errors) if errors else cls.from_ideal().readout
        return cls(epsilon=min(1.0, max(0.0, epsilon)), readout=readout, source=f"backend:{backend.name}",
                   line_qubits=metrics.line_qubits, log10_esp=metrics.log10_esp,
                   template=template_fingerprint(template))

    @classmethod
    def from_calibration(cls, readout_counts: Sequence[Sequence[int]],
                         known: Sequence[tuple[Sequence[str], Sequence[int]]] = (),
                         epsilon: float | None = None, source: str = "calibration",
                         line_qubits: tuple[int, ...] | None = None, template: dict | None = None) -> "NoiseProfile":
        """From calibration runs: readout_counts[i] are the counts after preparing line index i.

        ε and the bit factors are fitted from the single-win boards in `known`
        ((board, counts) pairs; see fit_line_signal), unless ε is given; the
        double-win boards in it set double_factor (fit_double_factor), and
        every board in it sets the losing lines' measured shares (losing_shares).
        """
        if len(readout_counts) != NUM_LINES:
            raise ValueError(f"need counts for all {NUM_LINES} prepared line indices")
        empty = [i for i, counts in enumerate(readout_counts) if sum(counts) <= 0]
        if empty:
            raise ValueError(f"readout circuits for line indices {empty} have no shots")
        columns = [[c / sum(counts) for c in counts] for counts in readout_counts]
        readout = tuple(tuple(columns[p][m] for p in range(NUM_LINES)) for m in range(NUM_LINES))
        factors, errors = (1.0, 1.0, 1.0), None
        fitted = epsilon is None
        if fitted:
            if not known:
                raise ValueError("pass known boards to fit epsilon, or epsilon itself")
            epsilon, factors, errors = fit_line_signal(readout, known)
        profile = cls(epsilon=epsilon, readout=readout, source=source, line_qubits=line_qubits, bit_factors=factors,
                      calibration_shots=sum(readout_counts[0]), signal_errors=errors, template=template)
        doubles = [(board, counts) for board, counts in known if len(_winners(board)) == 2]
        if fitted and doubles:
            profile = replace(profile, double_factor=fit_double_factor(profile, doubles))
        floor = losing_shares(profile, known)
        if floor is None:
            return profile
        shares, shots, ses, spread = floor
        return replace(profile, no_win_shares=shares, no_win_shots=shots, no_win_errors=ses, no_win_spread=spread)

    @classmethod
    def from_record(cls, saved: dict) -> "NoiseProfile":
        """A profile from its summary in a RunStore record (the "profile" field of most records).

        Fields added after the first records were saved fall back to their defaults.
        """
        def optional(key):
            return None if saved.get(key) is None else tuple(saved[key])

        return cls(epsilon=saved["epsilon"], readout=tuple(tuple(row) for row in saved["readout"]),
                   source=saved["source"],
                   line_qubits=optional("line_qubits"),
                   calibration_id=saved.get("calibration_id"),
                   bit_factors=tuple(saved.get("bit_factors") or (1.0, 1.0, 1.0)),
                   no_win_shares=optional("no_win_shares"),
                   no_win_shots=saved.get("no_win_shots") or 0,
                   no_win_errors=optional("no_win_errors"),
                   no_win_spread=optional("no_win_spread"),
                   double_factor=saved.get("double_factor", 1.0),
                   calibration_shots=saved.get("calibration_shots") or 0,
                   signal_errors=optional("signal_errors"),
                   template=saved.get("template"))

    def with_calibration_shots(self, shots: int) -> "NoiseProfile":
        """The profile a calibration with this many shots per circuit is expected to give.

        The fitted numbers stay; the standard errors of the measured losing
        shares shrink (or grow) with the shots. Unchanged without measured
        shares or a known calibration shot count.
        """
        if self.no_win_shares is None or not self.calibration_shots:
            return self
        scale = self.calibration_shots / shots
        errors = None if self.no_win_errors is None else tuple(e * sqrt(scale) for e in self.no_win_errors)
        return replace(self, no_win_shots=max(1, round(self.no_win_shots / scale)), no_win_errors=errors,
                       calibration_shots=shots)

    # --- the model ---

    def signal(self, line: int) -> float:
        """Share of a winning line's signal that survives: (1 − ε) times its bits' factors, at most 1."""
        kept = 1 - self.epsilon
        for b, factor in enumerate(self.bit_factors):
            if (line >> b) & 1:
                kept *= factor
        return min(1.0, kept)

    def predicted_for(self, winners: Sequence[int] | set[int]) -> tuple[float, ...]:
        """Measured line distribution for a board with these winning lines (see the module docstring)."""
        winners = set(winners)
        ideal = ideal_line_probabilities(winners)
        scale = self.double_factor if len(winners) > 1 else 1.0
        mixed = [0.0] * NUM_LINES
        for w in winners:
            s = min(1.0, scale * self.signal(w))
            mixed[w] = s * ideal[w] + (1 - s) / NUM_LINES
        rest = (1 - sum(mixed)) / (NUM_LINES - len(winners))
        mixed = [mixed[i] if i in winners else rest for i in range(NUM_LINES)]
        return tuple(sum(self.readout[m][p] * mixed[p] for p in range(NUM_LINES)) for m in range(NUM_LINES))

    @cached_property
    def q0(self) -> tuple[float, ...]:
        """Per line, the most a single shot can land on it when it is not a winner.

        Worst case over no win, every single win and every legal double win,
        so the false-flag budget holds on every board, not just on average. A
        profile with measured losing shares also takes, per line, the pooled
        mean share plus NO_WIN_MARGIN times √(SE² + spread²): the standard
        error of the mean and the spread between boards, so a board whose true
        share sits above the mean is covered too. The model spreads noise
        evenly over losing lines, and the device may not.
        """
        cases = [()] + [(w,) for w in range(NUM_LINES)] + list(legal_double_wins())
        worst = [0.0] * NUM_LINES
        for winners in cases:
            probs = self.predicted_for(winners)
            for i in range(NUM_LINES):
                if i not in winners:
                    worst[i] = max(worst[i], probs[i])
        if self.no_win_shares is not None:
            errors = self.no_win_errors or tuple(sqrt(q * (1 - q) / self.no_win_shots) for q in self.no_win_shares)
            spread = self.no_win_spread or (0.0,) * NUM_LINES
            worst = [max(w, q + NO_WIN_MARGIN * sqrt(se * se + tau * tau))
                     for w, q, se, tau in zip(worst, self.no_win_shares, errors, spread, strict=True)]
        return tuple(min(1 - _MIN_Q0, max(_MIN_Q0, q)) for q in worst)

    @cached_property
    def _single_win_shares(self) -> tuple[float, ...]:
        return tuple(self.predicted_for((w,))[w] for w in range(NUM_LINES))

    def cutoffs(self, shots: int, budget: float) -> tuple[int, ...]:
        """Per-line cutoffs: smallest k with P(K ≥ k | shots, q0ᵢ) ≤ budget / 8."""
        return tuple(binomial_cutoff(shots, budget, q) for q in self.q0)

    def miss_rates(self, shots: int, budget: float) -> MissRates:
        cutoffs = self.cutoffs(shots, budget)
        q0 = self.q0
        single = []
        undetectable = []
        for w in range(NUM_LINES):
            p = self._single_win_shares[w]
            single.append(binomial_below(cutoffs[w], shots, p))
            if p <= q0[w]:
                undetectable.append(w)
        double_both, double_any = {}, {}
        for a, b in legal_double_wins():
            probs = self.predicted_for((a, b))
            both = binomial_both_below(shots, cutoffs[a], probs[a], cutoffs[b], probs[b])
            double_both[(a, b)] = both
            double_any[(a, b)] = min(1.0, binomial_below(cutoffs[a], shots, probs[a])
                                     + binomial_below(cutoffs[b], shots, probs[b]) - both)
        no_win = self.no_win_shares if self.no_win_shares is not None else self.predicted_for(())
        false_flags = sum(1 - binomial_below(cutoffs[i], shots, no_win[i]) for i in range(NUM_LINES))
        return MissRates(shots, budget, cutoffs, tuple(single), double_both, double_any,
                         false_flags, tuple(undetectable))

    def shots_needed(self, target: float = 0.01, budget: float = DEFAULT_BUDGET, max_shots: int = 16384,
                     in_job: bool = False) -> int | None:
        """Fewest shots (to the nearest 64) that keep every win's miss chance at or below target.

        With in_job, the job is taken to calibrate at its own shot count, so
        the measured losing shares get more precise as the shots go up (see
        with_calibration_shots); otherwise this calibration's precision stays.
        None if even max_shots is not enough, which happens when noise leaves
        a line undetectable.
        """
        def ok(n):
            profile = self.with_calibration_shots(n) if in_job else self
            # Single wins are cheap to check and fail first on noisy backends.
            cutoffs = profile.cutoffs(n, budget)
            if any(binomial_below(cutoffs[w], n, profile._single_win_shares[w]) > target for w in range(NUM_LINES)):
                return False
            return profile.miss_rates(n, budget).worst <= target

        hi = 64
        while not ok(hi):
            if hi >= max_shots:
                return None
            hi = min(2 * hi, max_shots)
        lo = hi // 2
        while hi - lo > 64:
            mid = (lo + hi) // 2 // 64 * 64
            if ok(mid):
                hi = mid
            else:
                lo = mid
        return hi


def readout_matrix_from_errors(errors: Sequence[tuple[float, float]]) -> tuple[tuple[float, ...], ...]:
    """8×8 readout matrix from each line qubit's (P(read 1 | 0), P(read 0 | 1)), bit b = line_cl[b].

    Assumes the qubits' readout errors are independent; calibration runs
    measure the full matrix instead.
    """
    if len(errors) != 3:
        raise ValueError(f"need errors for the 3 line qubits, got {len(errors)}")
    per_bit = [((1 - p10, p01), (p10, 1 - p01)) for p10, p01 in errors]  # [measured][prepared]
    return tuple(
        tuple(_product(per_bit[b][(m >> b) & 1][(p >> b) & 1] for b in range(3)) for p in range(NUM_LINES))
        for m in range(NUM_LINES)
    )


def fit_epsilon(readout, known: Sequence[tuple[Sequence[str], Sequence[int]]]) -> float:
    """Least-squares ε for boards with known winners, given the readout matrix.

    The model is linear in s = 1 − ε: measured = A·u + s · A·(ideal − u).
    """
    num = den = 0.0
    uniform = ideal_line_probabilities(())
    for board, counts in known:
        winners = classical_winning_lines(board, X) | classical_winning_lines(board, O)
        ideal = ideal_line_probabilities(winners)
        shots = sum(counts)
        base = [sum(readout[m][p] * uniform[p] for p in range(NUM_LINES)) for m in range(NUM_LINES)]
        signal = [sum(readout[m][p] * (ideal[p] - uniform[p]) for p in range(NUM_LINES)) for m in range(NUM_LINES)]
        for m in range(NUM_LINES):
            num += signal[m] * (counts[m] / shots - base[m])
            den += signal[m] ** 2
    if den == 0:
        raise ValueError("known boards carry no signal: include at least one board with a winner")
    return min(1.0, max(0.0, 1 - num / den))


def _winners(board) -> set[int]:
    return classical_winning_lines(board, X) | classical_winning_lines(board, O)


def board_signal(readout, board, counts) -> float:
    """Least-squares share of signal s that survives on one board with known winners (see fit_epsilon)."""
    return 1 - fit_epsilon(readout, [(board, counts)])


def _board_signal_and_variance(readout, board, counts) -> tuple[float, float]:
    """board_signal and its shot-noise variance.

    s is linear in the measured shares, s = Σ g_m (share_m − base_m) / Σ g²,
    and the shares are multinomial, so Var(Σ g_m share_m) = (Σ g² p − (Σ g p)²) / n.
    """
    uniform = ideal_line_probabilities(())
    ideal = ideal_line_probabilities(_winners(board))
    g = [sum(readout[m][p] * (ideal[p] - uniform[p]) for p in range(NUM_LINES)) for m in range(NUM_LINES)]
    n = sum(counts)
    p = [(c + 0.5) / (n + NUM_LINES / 2) for c in counts]  # never exactly 0, so the variance never vanishes
    den = sum(x * x for x in g)
    var = (sum(x * x * q for x, q in zip(g, p, strict=True)) - sum(x * q for x, q in zip(g, p, strict=True)) ** 2)
    return board_signal(readout, board, counts), var / n / den ** 2


def fit_line_signal(readout, known: Sequence[tuple[Sequence[str], Sequence[int]]]) -> tuple[float, tuple, tuple]:
    """ε, the 3 bit factors and their standard errors, from boards with a single known winning line.

    Each board's surviving signal s and its shot-noise variance are measured
    (board_signal), then s = (1 − ε) · Π_b f_b^bit_b(line) is fitted by
    weighted least squares on s itself (Gauss-Newton), so a board where hardly
    any signal survives counts as much as its shot noise allows, no more. A bit
    that is 0 on every board's line keeps the factor 1: nothing measures it.
    When the mean signal is within SIGNAL_SIGMAS standard errors of 0, the bit
    factors cannot be told from noise and one ε is fitted; otherwise a factor
    within FACTOR_SIGMAS standard errors of 1 is set to 1 and the rest fitted
    again, so the factors kept are the ones the data measures. Boards with no or
    two winners are not used here. Returns (ε, factors, standard errors of
    (ε, f₀, f₁, f₂), 0 for a number not fitted).
    """
    rows = [(min(_winners(board)), *_board_signal_and_variance(readout, board, counts)) for board, counts in known
            if len(_winners(board)) == 1]
    if not rows:
        raise ValueError("known boards carry no signal: include at least one board with a single winner")
    lines = [line for line, _, _ in rows]
    signal = np.array([s for _, s, _ in rows])
    weight = 1 / np.array([v for _, _, v in rows])
    mean = float(np.dot(weight, signal) / weight.sum())
    bits = [b for b in range(3) if any((line >> b) & 1 for line in lines)]
    if mean < SIGNAL_SIGMAS / np.sqrt(weight.sum()):
        bits = []
    while True:
        theta, se = _fit_signal(lines, signal, weight, bits, mean)
        factors = {b: (float(np.exp(theta[i])), float(np.exp(theta[i]) * se[i])) for i, b in enumerate(bits, 1)}
        weak = [b for b, (f, e) in factors.items() if abs(f - 1) < FACTOR_SIGMAS * e]
        if not weak:
            break
        bits.remove(min(weak, key=lambda b: abs(factors[b][0] - 1) / factors[b][1]))
    kept = float(np.exp(theta[0]))
    values, errors = [1.0, 1.0, 1.0], [kept * float(se[0]), 0.0, 0.0, 0.0]
    for b, (f, e) in factors.items():
        values[b], errors[b + 1] = f, e
    return min(1.0, max(0.0, 1 - kept)), tuple(values), tuple(errors)


def _fit_signal(lines, signal, weight, bits, start):
    """Weighted Gauss-Newton fit of log(1 − ε) and log f_b for these bits; returns (θ, standard errors of θ)."""
    design = np.array([[1.0] + [float((line >> b) & 1) for b in bits] for line in lines])
    lower = np.array([np.log(_MIN_FACTOR)] * design.shape[1])
    upper = np.array([0.0] + [np.log(_MAX_FACTOR)] * len(bits))  # 1 − ε is at most 1
    theta = np.zeros(design.shape[1])
    theta[0] = np.log(min(1.0, max(_MIN_FACTOR, start)))

    def cost(t):
        return float(np.dot(weight, (signal - np.exp(design @ t)) ** 2))

    for _ in range(100):
        model = np.exp(design @ theta)
        jac = design * model[:, None]
        step, *_ = np.linalg.lstsq(jac * np.sqrt(weight)[:, None], (signal - model) * np.sqrt(weight), rcond=None)
        scale = 1.0
        while scale > 1e-6:  # halve the step until it does not make the fit worse
            trial = np.clip(theta + scale * step, lower, upper)
            if cost(trial) <= cost(theta):
                break
            scale /= 2
        else:
            break
        done = np.max(np.abs(trial - theta)) < 1e-12
        theta = trial
        if done:
            break
    model = np.exp(design @ theta)
    jac = design * model[:, None]
    cov = np.linalg.pinv(jac.T @ (jac * weight[:, None]))
    return theta, np.sqrt(np.maximum(np.diag(cov), 0.0))


def fit_double_factor(profile: "NoiseProfile", known: Sequence[tuple[Sequence[str], Sequence[int]]]) -> float:
    """Least-squares double_factor from boards with two known winning lines, given the rest of the profile.

    Each winning line w of a double win gets d · s_w · (ideal_w − 1/8) above
    1/8, the losing lines share what is left, and the measured shares are A
    times that: linear in d, as fit_epsilon is in s.
    """
    num = den = 0.0
    uniform = ideal_line_probabilities(())
    for board, counts in known:
        winners = _winners(board)
        ideal = ideal_line_probabilities(winners)
        h = [profile.signal(w) * (ideal[w] - uniform[w]) if w in winners else 0.0 for w in range(NUM_LINES)]
        rest = -sum(h) / (NUM_LINES - len(winners))
        h = [h[i] if i in winners else rest for i in range(NUM_LINES)]
        base = [sum(profile.readout[m][p] * uniform[p] for p in range(NUM_LINES)) for m in range(NUM_LINES)]
        g = [sum(profile.readout[m][p] * h[p] for p in range(NUM_LINES)) for m in range(NUM_LINES)]
        shots = sum(counts)
        for m in range(NUM_LINES):
            num += g[m] * (counts[m] / shots - base[m])
            den += g[m] ** 2
    if den == 0:
        return 1.0  # no signal survives on single wins either: nothing to scale
    return min(_MAX_FACTOR, max(0.0, num / den))


def losing_shares(profile: "NoiseProfile", known: Sequence[tuple[Sequence[str], Sequence[int]]]):
    """Each line's measured share when it loses, pooled over the known boards, as a no-win board's share.

    On a board with winners W the model predicts a losing line i at
    pred_W[i]; its measured share there, times pred_none[i] / pred_W[i], is
    what the line would get on a board without a winner. Two estimates are
    made per line: from the no-win boards alone (no model involved), and
    pooled over every board where i loses (the single and double wins through
    the model ratio, so the 2 no-win boards are not the only measurement).
    Each weights its boards by their shot-noise variance. The spread between
    boards beyond shot noise is estimated over every board (DerSimonian-Laird;
    two no-win boards alone cannot measure it). The line keeps whichever mean
    ends higher (mean + NO_WIN_MARGIN · √(SE² + spread²)), so the single wins
    can raise a floor but never lower it below what the no-win boards measured.
    Returns (shares, total shots of the boards used, standard errors, spread),
    or None without known boards.
    """
    none = profile.predicted_for(())
    rows = []  # (line, no-win board?, share as a no-win board's, its variance)
    for board, counts in known:
        winners = _winners(board)
        pred = profile.predicted_for(winners)
        n = sum(counts)
        for i in range(NUM_LINES):
            if i in winners or pred[i] <= 0:
                continue
            ratio = none[i] / pred[i]
            p = (counts[i] + 0.5) / (n + 1)  # never exactly 0, so the variance never vanishes
            rows.append((i, not winners, ratio * counts[i] / n, ratio * ratio * p * (1 - p) / n))
    if not rows:
        return None

    def pooled(values, variances):
        x, u = np.array(values), 1 / np.array(variances)
        mean = float(np.dot(u, x) / u.sum())
        q = float(np.dot(u, (x - mean) ** 2))
        tau2 = max(0.0, (q - (len(x) - 1)) / (u.sum() - np.dot(u, u) / u.sum())) if len(x) > 1 else 0.0
        return mean, float(1 / np.sqrt(u.sum())), sqrt(tau2)

    shares, errors, spread = [], [], []
    for i in range(NUM_LINES):
        every = [r for r in rows if r[0] == i]
        if not every:
            return None
        mean, se, tau = pooled([r[2] for r in every], [r[3] for r in every])
        direct = [r for r in every if r[1]]
        if direct:
            # The spread comes from every board: two no-win boards alone cannot measure it.
            d_mean, d_se, _ = pooled([r[2] for r in direct], [r[3] for r in direct])
            if d_mean + NO_WIN_MARGIN * sqrt(d_se ** 2 + tau ** 2) > mean + NO_WIN_MARGIN * sqrt(se ** 2 + tau ** 2):
                mean, se = d_mean, d_se
        shares.append(mean)
        errors.append(se)
        spread.append(tau)
    return tuple(shares), sum(sum(c) for _, c in known), tuple(errors), tuple(spread)


def known_boards(variant: str = "player") -> tuple:
    """The calibration boards as (board, player) pairs for this variant (the legacy full variant takes no player)."""
    return CALIBRATION_BOARDS if variant == "player" else tuple((board, None) for board, _ in CALIBRATION_BOARDS)


CALIBRATION_CIRCUITS = NUM_LINES + len(CALIBRATION_BOARDS)  # readout circuits plus the known boards, per job


def calibration_circuits(backend: Backend, variant: str = "player",
                         known: Sequence[tuple[Sequence[str], str]] = CALIBRATION_BOARDS):
    """The circuits a calibration needs, transpiled for the backend, and the line qubits they measure.

    The 8 readout circuits prepare line index 0-7 directly on the search
    circuit's line qubits; then one search per known (board, player) pair.
    """
    line_qubits = backend.metrics(backend.search_template(variant)).line_qubits
    circuits = backend.readout_circuits(line_qubits)
    circuits += [backend.search_circuit(variant, board, player) for board, player in known]
    return circuits, line_qubits


def profile_from_calibration(backend: Backend, counts: Sequence[Sequence[int]], line_qubits,
                             known: Sequence[tuple[Sequence[str], str]] = CALIBRATION_BOARDS,
                             variant: str = "player") -> "NoiseProfile":
    """Fit a profile from the counts of calibration_circuits(), in the same order."""
    return NoiseProfile.from_calibration(
        counts[:NUM_LINES],
        known=[(board, c) for (board, _), c in zip(known, counts[NUM_LINES:NUM_LINES + len(known)], strict=True)],
        source=f"calibration:{backend.name}",
        line_qubits=line_qubits,
        template=template_fingerprint(backend.search_template(variant)),
    )


def calibrate(backend: Backend, variant: str = "player", shots: int = 1024,
              known: Sequence[tuple[Sequence[str], str]] = CALIBRATION_BOARDS):
    """Measure a profile on the backend itself, in one job.

    Runs the 8 readout circuits and the search on `known` (board, player)
    pairs, then fits the readout matrix, ε, the bit factors and the no-win
    shares. Works on any provider.
    detect_winners does this inside its own job by default; use calibrate on
    its own to check a backend before running anything else. Returns
    (profile, job output).
    """
    circuits, line_qubits = calibration_circuits(backend, variant, known)
    out = backend.run(circuits, shots)
    return profile_from_calibration(backend, out.counts, line_qubits, known, variant), out


def _product(values):
    result = 1.0
    for v in values:
        result *= v
    return result
