"""Binomial helpers against brute-force sums."""

import math
from itertools import product

import pytest

from qttt.stats import binomial_below, binomial_both_below


def _pmf(j, n, q):
    return math.comb(n, j) * q ** j * (1 - q) ** (n - j)


@pytest.mark.parametrize(("k", "n", "q"), [(0, 10, 0.3), (3, 10, 0.3), (11, 10, 0.3), (5, 20, 0.0), (5, 20, 1.0),
                                           (153, 1024, 0.125), (153, 1024, 0.5)])
def test_binomial_below_matches_direct_sum(k, n, q):
    direct = sum(_pmf(j, n, q) for j in range(min(k, n + 1)))
    assert binomial_below(k, n, q) == pytest.approx(direct, abs=1e-12)


def _both_below_brute_force(n, k1, p1, k2, p2):
    total = 0.0
    for a, b in product(range(n + 1), repeat=2):
        if a + b <= n and a < k1 and b < k2:
            rest = n - a - b
            total += (math.factorial(n) / (math.factorial(a) * math.factorial(b) * math.factorial(rest))
                      * p1 ** a * p2 ** b * (1 - p1 - p2) ** rest)
    return total


@pytest.mark.parametrize(("n", "k1", "p1", "k2", "p2"), [(12, 4, 0.3, 5, 0.25), (20, 6, 0.4, 6, 0.4),
                                                         (15, 3, 0.1, 9, 0.6), (10, 11, 0.5, 11, 0.5)])
def test_both_below_matches_multinomial(n, k1, p1, k2, p2):
    expected = _both_below_brute_force(n, k1, p1, k2, p2)
    assert binomial_both_below(n, k1, p1, k2, p2) == pytest.approx(expected, abs=1e-12)


def test_both_below_shared_shots_are_negatively_correlated():
    # Shots that land on one line cannot land on the other, so the two counts
    # move in opposite directions: both low together is less likely than for
    # independent counts.
    n, k, p = 1024, 153, 0.1375
    independent = binomial_below(k, n, p) ** 2
    assert binomial_both_below(n, k, p, k, p) < independent
