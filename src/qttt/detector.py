"""Winner detector: one Grover search, a check-budget cutoff, and classical checks of flagged lines."""

from dataclasses import dataclass
from math import exp, lgamma, log

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2

from qttt.board import EMPTY, WIN_LINES
from qttt.encoding import bitstring_to_line_index
from qttt.grover import build_search_circuit

DEFAULT_SHOTS = 1024
DEFAULT_BUDGET = 0.1
NUM_LINES = len(WIN_LINES)

def _binomial_log_pmf(j, n, q):
    """log P(K = j) for K ~ Binomial(n, q)"""
    return lgamma(n+1) - lgamma(j+1) - lgamma(n-j+1) + j*log(q) + (n-j)*log(1-q)

def _binomial_tail(k, n, q):
    """P(K >= k)"""
    if k <= 0: return 1.0 # Every count is at least 0
    if k > n: return 0.0 # A count cannot exceed the shot count

    ret = 0.0
    for j in range(k, n+1):
        ret += exp(_binomial_log_pmf(j, n, q))
    return ret

def binomial_cutoff(shots=DEFAULT_SHOTS, budget=DEFAULT_BUDGET, q0=1/8) -> int:
    """
    returns the smallest count k where the expected number of falsely flagged lines,
    8 · P(K ≥ k | shots, q0), is at most budget, and a line is flagged when count >= k.
    """
    if shots < 1: raise ValueError(f"shots must be at least 1, got {shots}")
    if not 0 < budget <= NUM_LINES: raise ValueError(f"budget must be in (0, {NUM_LINES}], got {budget}")
    if not 0 < q0 < 1: raise ValueError(f"q0 must be in (0, 1), got {q0}")

    # Walk down from k = shots + 1 (tail = 0), adding one term per step, and
    # stop before the tail goes over budget. O(shots) instead of O(shots²).
    k = shots + 1
    tail = 0.0
    while k > 0:
        next_tail = tail + exp(_binomial_log_pmf(k-1, shots, q0))
        if NUM_LINES * next_tail > budget: break
        tail = next_tail
        k -= 1
    return k

@dataclass(frozen=True)
class SearchRun:
    """One run of the search circuit: counts per line index."""
    shots: int
    counts: tuple[int, ...]
    circuit: QuantumCircuit

    @property
    def probabilities(self) -> tuple[float, ...]:
        return tuple(c / self.shots for c in self.counts)

@dataclass(frozen=True)
class DetectionResult:
    """Detector verdict for one board, plus the run behind it."""
    board: tuple[str, ...]
    variant: str
    winner: str | None
    lines: frozenset[int]
    candidates: tuple[int, ...]
    rejected: tuple[int, ...]
    cutoff: int
    budget: float
    run: SearchRun

    @property
    def probabilities(self) -> tuple[float, ...]:
        return self.run.probabilities
    
    @property
    def checks(self) -> int:
        return len(self.candidates)

def run_search(board, variant="full", player=None, shots=DEFAULT_SHOTS, sampler=None) -> SearchRun:
    """Run the search circuit once and count how often each line index was measured."""
    qc = build_search_circuit(variant, board, player)
    if sampler is None:
        sampler = SamplerV2()
    tqc = transpile(qc, AerSimulator())
    result = sampler.run([tqc], shots=shots).result()[0]
    raw = result.data.line_cl.get_counts()

    counts = [0] * NUM_LINES
    for bits, n in raw.items():
        counts[bitstring_to_line_index(bits)] = n
    return SearchRun(shots, tuple(counts), qc)

def _flag_and_check(board, counts, cutoff):
    """Flag lines with count >= cutoff and confirm each one on the board.

    Returns (candidates, rejected, lines, winner).
    """

    candidates = tuple(i for i in range(NUM_LINES) if counts[i] >= cutoff)
    confirmed = []
    rejected = []
    for i in candidates:
        a, b, c = WIN_LINES[i]
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            confirmed.append(i)
        else:
            rejected.append(i)
    lines = frozenset(confirmed)
    rejected = tuple(rejected)

    winner = board[WIN_LINES[min(lines)][0]] if lines else None
    return candidates, rejected, lines, winner

def detect_winner(board, variant="full", player=None, shots=DEFAULT_SHOTS,
                  budget=DEFAULT_BUDGET, sampler=None) -> DetectionResult:
    """Detect whether anyone has won, using one Grover search plus classical checks.

    Lines measured at least `cutoff` times are checked on the board; any line
    reported in the result is a real winning line.
    """
    cutoff = binomial_cutoff(shots, budget)
    run = run_search(board, variant, player, shots, sampler)
    candidates, rejected, lines, winner = _flag_and_check(board, run.counts, cutoff)
    return DetectionResult(
        board=tuple(board),
        variant=variant,
        winner=winner,
        lines=lines,
        candidates=candidates,
        rejected=rejected,
        cutoff=cutoff,
        budget=budget,
        run=run,
    )
