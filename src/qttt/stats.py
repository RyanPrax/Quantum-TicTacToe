"""Binomial math for the check-budget cutoff and miss-rate predictions."""

from math import exp, lgamma, log, sqrt

from qttt.board import WIN_LINES

NUM_LINES = len(WIN_LINES)


def _binomial_log_pmf(j, n, q):
    """log P(K = j) for K ~ Binomial(n, q)"""
    return lgamma(n+1) - lgamma(j+1) - lgamma(n-j+1) + j*log(q) + (n-j)*log(1-q)


def _binomial_pmf(j, n, q):
    """P(K = j), also when q is exactly 0 or 1."""
    if not 0 <= j <= n: return 0.0
    if q <= 0: return 1.0 if j == 0 else 0.0
    if q >= 1: return 1.0 if j == n else 0.0
    return exp(_binomial_log_pmf(j, n, q))


def _binomial_tail(k, n, q):
    """P(K >= k)"""
    if k <= 0: return 1.0 # Every count is at least 0
    if k > n: return 0.0 # A count cannot exceed the shot count

    ret = 0.0
    for j in range(k, n+1):
        ret += exp(_binomial_log_pmf(j, n, q))
    return ret


def _central_range(n, q):
    """Counts within 12 standard deviations of the mean; terms outside add less than 1e-30."""
    if not 0 < q < 1:
        return 0, n
    mean, spread = n * q, 12 * sqrt(n * q * (1 - q)) + 1
    return max(0, int(mean - spread)), min(n, int(mean + spread) + 1)


def binomial_below(k, n, q):
    """P(K < k) for K ~ Binomial(n, q): the chance a line with per-shot probability q is not flagged."""
    lo, hi = _central_range(n, q)
    return min(1.0, sum(_binomial_pmf(j, n, q) for j in range(lo, min(k, hi + 1))))


def binomial_both_below(n, k1, p1, k2, p2):
    """P(K1 < k1 and K2 < k2) when two lines share the same n shots.

    (K1, K2, rest) is multinomial, so K2 given K1 = a is Binomial(n - a, p2 / (1 - p1)).
    """
    if p1 >= 1:
        return 1.0 if n < k1 and k2 > 0 else 0.0
    c = min(1.0, p2 / (1 - p1))
    lo, hi = _central_range(n, p1)
    total = 0.0
    for a in range(lo, min(k1, hi + 1)):
        total += _binomial_pmf(a, n, p1) * binomial_below(k2, n - a, c)
    return min(1.0, total)


def binomial_cutoff(shots=1024, budget=0.1, q0=1/8) -> int:
    """
    returns the smallest count k where the expected number of falsely flagged lines,
    8 · P(K ≥ k | shots, q0), is at most budget, and a line is flagged when count >= k.

    With a per-line q0ᵢ this is also the per-line cutoff: P(K ≥ k | shots, q0ᵢ) ≤ budget / 8.
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
