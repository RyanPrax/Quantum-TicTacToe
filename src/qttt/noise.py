"""Noise profiles: per-line cutoffs and predicted miss rates for a backend.

A profile models what one round measures as

    measured = A · [(1 − ε) · ideal + ε / 8]

where ideal is the noiseless line distribution for the board, ε is the share of
shots that gate noise scatters evenly, and A[measured][prepared] is the 8×8
readout matrix of the 3 line qubits. From that it gives each line its own
cutoff (lines that noise pushes up need more counts to be flagged) and
predicts how often wins are missed before any hardware time is spent.

Profiles come from no noise (from_ideal), from a backend's published
calibration data (from_backend), or from calibration runs on the backend
itself (calibrate), which works on any provider.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations
from math import asin, sin, sqrt

from qttt.backends import Backend
from qttt.board import EMPTY, O, X, boards_by_win_count, classical_winning_lines
from qttt.stats import NUM_LINES, binomial_below, binomial_both_below, binomial_cutoff

_MIN_Q0 = 1e-12  # keeps binomial_cutoff's q0 strictly inside (0, 1)

# A legal single-win board, used to measure ε when calibrating.
CALIBRATION_BOARD = (X, X, X,
                     O, O, EMPTY,
                     EMPTY, EMPTY, EMPTY)


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
    false_flags_no_win: float  # expected rejected checks on a board with no winner
    undetectable: tuple[int, ...]  # lines whose winner looks no more likely than a loser

    @property
    def worst(self) -> float:
        """The largest chance of missing a win, over every single and double win."""
        return max([*self.single, *self.double_both.values()])


@dataclass(frozen=True)
class NoiseProfile:
    """Gate noise ε and readout matrix A for the 3 line qubits of one backend."""
    epsilon: float
    readout: tuple[tuple[float, ...], ...]  # A[measured][prepared]
    source: str
    line_qubits: tuple[int, ...] | None = None  # physical qubits the readout matrix belongs to
    log10_esp: float | None = field(default=None, compare=False)

    def __post_init__(self):
        if not 0 <= self.epsilon <= 1:
            raise ValueError(f"epsilon must be in [0, 1], got {self.epsilon}")
        if len(self.readout) != NUM_LINES or any(len(row) != NUM_LINES for row in self.readout):
            raise ValueError("readout must be an 8x8 matrix")

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
        metrics = backend.metrics(backend.search_template(variant))
        epsilon = 0.0 if metrics.log10_esp is None else 1 - 10 ** metrics.log10_esp
        errors = backend.readout_errors(metrics.line_qubits)
        readout = readout_matrix_from_errors(errors) if errors else cls.from_ideal().readout
        return cls(epsilon=min(1.0, max(0.0, epsilon)), readout=readout, source=f"backend:{backend.name}",
                   line_qubits=metrics.line_qubits, log10_esp=metrics.log10_esp)

    @classmethod
    def from_calibration(cls, readout_counts: Sequence[Sequence[int]],
                         known: Sequence[tuple[Sequence[str], Sequence[int]]] = (),
                         epsilon: float | None = None, source: str = "calibration",
                         line_qubits: tuple[int, ...] | None = None) -> "NoiseProfile":
        """From calibration runs: readout_counts[i] are the counts after preparing line index i.

        ε is fitted from `known` (board, counts) pairs, boards whose winning
        lines are known, unless given directly.
        """
        if len(readout_counts) != NUM_LINES:
            raise ValueError(f"need counts for all {NUM_LINES} prepared line indices")
        columns = [[c / sum(counts) for c in counts] for counts in readout_counts]
        readout = tuple(tuple(columns[p][m] for p in range(NUM_LINES)) for m in range(NUM_LINES))
        if epsilon is None:
            if not known:
                raise ValueError("pass known boards to fit epsilon, or epsilon itself")
            epsilon = fit_epsilon(readout, known)
        return cls(epsilon=epsilon, readout=readout, source=source, line_qubits=line_qubits)

    # --- the model ---

    def predicted(self, ideal: Sequence[float]) -> tuple[float, ...]:
        """Measured line distribution for an ideal one: A · [(1 − ε) · ideal + ε / 8]."""
        mixed = [(1 - self.epsilon) * p + self.epsilon / NUM_LINES for p in ideal]
        return tuple(sum(self.readout[m][p] * mixed[p] for p in range(NUM_LINES)) for m in range(NUM_LINES))

    def predicted_for(self, winners: Sequence[int] | set[int]) -> tuple[float, ...]:
        return self.predicted(ideal_line_probabilities(winners))

    @property
    def q0(self) -> tuple[float, ...]:
        """Per line, the most a single shot can land on it when it is not a winner.

        Worst case over no win, every single win and every legal double win,
        so the false-flag budget holds on every board, not just on average.
        """
        cases = [()] + [(w,) for w in range(NUM_LINES)] + list(legal_double_wins())
        worst = [0.0] * NUM_LINES
        for winners in cases:
            probs = self.predicted_for(winners)
            for i in range(NUM_LINES):
                if i not in winners:
                    worst[i] = max(worst[i], probs[i])
        return tuple(min(1 - _MIN_Q0, max(_MIN_Q0, q)) for q in worst)

    def cutoffs(self, shots: int, budget: float) -> tuple[int, ...]:
        """Per-line cutoffs: smallest k with P(K ≥ k | shots, q0ᵢ) ≤ budget / 8."""
        return tuple(binomial_cutoff(shots, budget, q) for q in self.q0)

    def miss_rates(self, shots: int, budget: float) -> MissRates:
        cutoffs = self.cutoffs(shots, budget)
        q0 = self.q0
        single = []
        undetectable = []
        for w in range(NUM_LINES):
            p = self.predicted_for((w,))[w]
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
        no_win = self.predicted_for(())
        false_flags = sum(1 - binomial_below(cutoffs[i], shots, no_win[i]) for i in range(NUM_LINES))
        return MissRates(shots, budget, cutoffs, tuple(single), double_both, double_any,
                         false_flags, tuple(undetectable))

    def shots_needed(self, target: float = 0.01, budget: float = 0.1, max_shots: int = 16384) -> int | None:
        """Fewest shots (to the nearest 64) that keep every win's miss chance at or below target.

        None if even max_shots is not enough, which happens when noise leaves
        a line undetectable.
        """
        def ok(n):
            # Single wins are cheap to check and fail first on noisy backends.
            cutoffs = self.cutoffs(n, budget)
            if any(binomial_below(cutoffs[w], n, self.predicted_for((w,))[w]) > target for w in range(NUM_LINES)):
                return False
            return self.miss_rates(n, budget).worst <= target

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


DEFAULT_KNOWN = ((CALIBRATION_BOARD, X),)


def calibration_circuits(backend: Backend, variant: str = "player",
                         known: Sequence[tuple[Sequence[str], str]] = DEFAULT_KNOWN):
    """The circuits a calibration needs, transpiled for the backend, and the line qubits they measure.

    The 8 readout circuits prepare line index 0-7 directly on the search
    circuit's line qubits; then one search per known (board, player) pair.
    """
    line_qubits = backend.metrics(backend.search_template(variant)).line_qubits
    circuits = backend.readout_circuits(line_qubits)
    circuits += [backend.search_circuit(variant, board, player) for board, player in known]
    return circuits, line_qubits


def profile_from_calibration(backend: Backend, counts: Sequence[Sequence[int]], line_qubits,
                             known: Sequence[tuple[Sequence[str], str]] = DEFAULT_KNOWN) -> "NoiseProfile":
    """Fit a profile from the counts of calibration_circuits(), in the same order."""
    return NoiseProfile.from_calibration(
        counts[:NUM_LINES],
        known=[(board, c) for (board, _), c in zip(known, counts[NUM_LINES:NUM_LINES + len(known)])],
        source=f"calibration:{backend.name}",
        line_qubits=line_qubits,
    )


def calibrate(backend: Backend, variant: str = "player", shots: int = 1024,
              known: Sequence[tuple[Sequence[str], str]] = DEFAULT_KNOWN):
    """Measure a profile on the backend itself, in one job.

    Runs the 8 readout circuits and the search on `known` (board, player)
    pairs, then fits the readout matrix and ε. Works on any provider.
    detect_winners does this inside its own job by default; use calibrate on
    its own to check a backend before running anything else. Returns
    (profile, job output).
    """
    circuits, line_qubits = calibration_circuits(backend, variant, known)
    out = backend.run(circuits, shots)
    return profile_from_calibration(backend, out.counts, line_qubits, known), out


def _product(values):
    result = 1.0
    for v in values:
        result *= v
    return result
