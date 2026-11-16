"""Noise profiles: model, per-line cutoffs, miss predictions and calibration."""

import math

import pytest

from qttt.backends import IdealBackend, get_backend
from qttt.board import O, X, boards_by_win_count, classical_winning_lines
from qttt.noise import (
    CALIBRATION_CIRCUITS,
    NoiseProfile,
    calibrate,
    fit_epsilon,
    ideal_line_probabilities,
    legal_double_wins,
    readout_matrix_from_errors,
)
from qttt.stats import DEFAULT_BUDGET, NUM_LINES, binomial_cutoff

IDENTITY = NoiseProfile.from_ideal().readout


def _profile(epsilon, readout=IDENTITY):
    return NoiseProfile(epsilon=epsilon, readout=readout, source="test")


# --- ideal model ---

def test_ideal_line_probabilities():
    assert ideal_line_probabilities(()) == pytest.approx([1 / 8] * 8)
    assert ideal_line_probabilities({3}) == pytest.approx([25 / 32 if i == 3 else 1 / 32 for i in range(8)])
    assert ideal_line_probabilities({0, 3}) == pytest.approx([0.5 if i in (0, 3) else 0.0 for i in range(8)])


def test_legal_double_wins_come_from_real_boards():
    pairs = set(legal_double_wins())
    from_boards = {tuple(sorted(classical_winning_lines(b, X))) for b in boards_by_win_count()[2]}
    assert pairs == from_boards


# --- readout matrix ---

def test_readout_matrix_columns_sum_to_one():
    A = readout_matrix_from_errors([(0.01, 0.05), (0.02, 0.03), (0.0, 0.1)])
    for prepared in range(NUM_LINES):
        assert sum(A[m][prepared] for m in range(NUM_LINES)) == pytest.approx(1.0)


def test_readout_matrix_without_errors_is_identity():
    assert readout_matrix_from_errors([(0.0, 0.0)] * 3) == IDENTITY


def test_readout_matrix_single_bit_flip():
    # Only line_cl[0] can flip, 1 -> 0 with probability 0.2: index 1 is read as 0 one time in five.
    A = readout_matrix_from_errors([(0.0, 0.2), (0.0, 0.0), (0.0, 0.0)])
    assert A[0][1] == pytest.approx(0.2)
    assert A[1][1] == pytest.approx(0.8)
    assert A[0][0] == pytest.approx(1.0)


def test_predicted_is_a_distribution():
    profile = _profile(0.4, readout_matrix_from_errors([(0.02, 0.05)] * 3))
    for winners in [(), (2,), (0, 3)]:
        assert sum(profile.predicted_for(winners)) == pytest.approx(1.0)


# --- cutoffs and miss rates ---

BUDGETS = (0.1, DEFAULT_BUDGET)


@pytest.mark.parametrize("budget", BUDGETS)
def test_ideal_profile_keeps_the_old_cutoff(budget):
    profile = NoiseProfile.from_ideal()
    assert profile.q0 == pytest.approx((1 / 8,) * 8)
    for shots in (100, 1024, 4096):
        assert profile.cutoffs(shots, budget) == (binomial_cutoff(shots, budget),) * 8


@pytest.mark.parametrize("budget", BUDGETS)
def test_readout_bias_raises_cutoffs_on_lines_it_pushes_up(budget):
    # Every qubit reads 1 as 0 often: shots drift toward index 0 (all bits 0) and away from 7.
    profile = _profile(0.5, readout_matrix_from_errors([(0.0, 0.1)] * 3))
    q0 = profile.q0
    assert q0[0] > 1 / 8 > q0[7]
    cutoffs = profile.cutoffs(1024, budget)
    assert cutoffs[0] > cutoffs[7]


@pytest.mark.parametrize("budget", BUDGETS)
def test_per_line_budget_holds_on_a_no_win_board(budget):
    profile = _profile(0.5, readout_matrix_from_errors([(0.01, 0.08), (0.03, 0.02), (0.0, 0.05)]))
    assert profile.miss_rates(1024, budget).false_flags_no_win <= budget


def _log_pmf(j, n, q):
    log_comb = math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
    return log_comb + j * math.log(q) + (n - j) * math.log(1 - q)


def test_miss_rates_at_the_default_budget_match_a_direct_sum():
    # Uniform noise ε = 0.9: no losing line ever beats 1/8 (single wins leave 0.116, double wins 0.1125),
    # so q0 = 1/8 everywhere, and a single win sits at 0.1 · 25/32 + 0.9 / 8.
    shots, p_win = 1024, 0.1 * 25 / 32 + 0.9 / 8
    m = _profile(0.9).miss_rates(shots, DEFAULT_BUDGET)
    tail = [sum(math.exp(_log_pmf(j, shots, 1 / 8)) for j in range(k, shots + 1)) for k in range(shots + 2)]
    k = min(k for k in range(shots + 2) if 8 * tail[k] <= DEFAULT_BUDGET)
    assert m.cutoffs == (k,) * 8
    assert m.single[0] == pytest.approx(sum(math.exp(_log_pmf(j, shots, p_win)) for j in range(k)), rel=1e-9)
    assert m.false_flags_no_win == pytest.approx(8 * tail[k], rel=1e-9)
    assert m.false_flags_no_win <= DEFAULT_BUDGET


def test_miss_rates_match_the_design_doc():
    # Doc section 5 at 1024 shots, B = 0.1, uniform noise.
    m9 = _profile(0.9).miss_rates(1024, 0.1)
    assert m9.single[0] == pytest.approx(2.38e-4, rel=0.01)
    assert max(m9.double_both.values()) == pytest.approx(0.00766, rel=0.01)
    assert max(m9.double_any.values()) == pytest.approx(0.230, rel=0.01)
    assert NoiseProfile.from_ideal().miss_rates(1024, 0.1).worst < 1e-12


def test_pure_noise_is_undetectable():
    m = _profile(1.0).miss_rates(1024, 0.1)
    assert m.undetectable == tuple(range(8))
    assert _profile(1.0).shots_needed(0.01, max_shots=1024) is None


@pytest.mark.parametrize("budget", BUDGETS)
def test_shots_needed_is_the_smallest_multiple_of_64(budget):
    profile = _profile(0.9)
    shots = profile.shots_needed(0.01, budget)
    assert shots % 64 == 0
    assert profile.miss_rates(shots, budget).worst <= 0.01
    assert profile.miss_rates(shots - 64, budget).worst > 0.01


def test_shots_needed_defaults_to_the_detectors_budget():
    assert _profile(0.9).shots_needed(0.01) == _profile(0.9).shots_needed(0.01, DEFAULT_BUDGET)


def test_profile_rejects_bad_values():
    with pytest.raises(ValueError):
        _profile(1.5)
    with pytest.raises(ValueError):
        NoiseProfile(epsilon=0.1, readout=((1.0,),), source="test")


# --- calibration ---

def _expected_counts(profile, winners, shots=10**6):
    return [round(p * shots) for p in profile.predicted_for(winners)]


def test_from_calibration_recovers_a_known_profile():
    true = _profile(0.37, readout_matrix_from_errors([(0.01, 0.04), (0.02, 0.03), (0.015, 0.05)]))
    readout_counts = [[round(true.readout[m][p] * 10**6) for m in range(8)] for p in range(8)]
    board = (X, X, X, O, O, " ", " ", " ", " ")
    fitted = NoiseProfile.from_calibration(readout_counts, known=[(board, _expected_counts(true, {0}))])
    assert fitted.epsilon == pytest.approx(0.37, abs=1e-4)
    for m in range(8):
        assert fitted.readout[m] == pytest.approx(true.readout[m], abs=1e-5)  # counts are rounded


def test_fit_epsilon_needs_a_winner():
    no_win = (X, O, X, X, O, O, O, X, X)
    with pytest.raises(ValueError):
        fit_epsilon(IDENTITY, [(no_win, [128] * 8)])
    with pytest.raises(ValueError):
        NoiseProfile.from_calibration([[1] * 8] * 8)


def test_calibrate_on_ideal_backend():
    profile, out = calibrate(IdealBackend(seed=4903), shots=2048)
    assert len(out.counts) == CALIBRATION_CIRCUITS
    for i in range(NUM_LINES):
        assert profile.readout[i][i] == pytest.approx(1.0)
    assert profile.epsilon == pytest.approx(0.0, abs=0.05)


# --- published calibration data (fake backend, transpile only) ---

def test_from_backend_uses_published_data():
    backend = get_backend("fake_berlin", seed=4903)
    profile = NoiseProfile.from_backend(backend)
    assert 0 < profile.epsilon < 1
    assert profile.line_qubits == backend.metrics(backend.search_template("player")).line_qubits
    assert profile.readout != IDENTITY


# --- signal per line: bit factors and measured no-win shares ---

def test_all_bit_factors_one_is_the_even_scatter_model():
    profile = _profile(0.3, readout_matrix_from_errors([(0.02, 0.05)] * 3))
    for winners in [(), (6,), (0, 3)]:
        ideal = ideal_line_probabilities(winners)
        mixed = [0.7 * p + 0.3 / 8 for p in ideal]
        old = [sum(profile.readout[m][p] * mixed[p] for p in range(8)) for m in range(8)]
        assert profile.predicted_for(winners) == pytest.approx(old)


def test_bit_factors_lower_the_signal_of_lines_with_that_bit():
    profile = NoiseProfile(0.4, IDENTITY, "test", bit_factors=(1.0, 1.0, 0.5))
    assert profile.signal(3) == pytest.approx(0.6)  # 011: bit 2 is 0
    assert profile.signal(4) == pytest.approx(0.3)  # 100
    assert profile.predicted_for({4})[4] < profile.predicted_for({3})[3]
    assert sum(profile.predicted_for({4, 7})) == pytest.approx(1.0)


def test_calibration_recovers_epsilon_and_bit_factors():
    from qttt.noise import CALIBRATION_BOARDS

    truth = NoiseProfile(0.4, IDENTITY, "truth", bit_factors=(0.95, 0.9, 0.8))
    known = [(board, _expected_counts(truth, classical_winning_lines(board, player)))
             for board, player in CALIBRATION_BOARDS]
    readout_counts = [[10**6 if m == p else 0 for m in range(8)] for p in range(8)]
    fitted = NoiseProfile.from_calibration(readout_counts, known)
    assert fitted.epsilon == pytest.approx(0.4, abs=1e-4)
    assert fitted.bit_factors == pytest.approx((0.95, 0.9, 0.8), abs=1e-4)
    assert fitted.double_factor == pytest.approx(1.0, abs=1e-3)
    assert fitted.no_win_shots == sum(sum(c) for _, c in known)  # every board where a line loses, pooled
    assert fitted.no_win_shares == pytest.approx([1 / 8] * 8, abs=1e-5)
    assert fitted.no_win_spread == pytest.approx([0.0] * 8, abs=1e-4)
    assert all(0 < e < 0.01 for e in fitted.signal_errors)
    assert fitted.calibration_shots == 10**6


def _calibration_counts(truth, shots, seed):
    """Readout counts and known boards measured on a device that behaves exactly like truth, with shot noise."""
    import numpy as np

    from qttt.noise import CALIBRATION_BOARDS

    rng = np.random.default_rng(seed)
    readout_counts = [list(rng.multinomial(shots, [truth.readout[m][p] for m in range(8)])) for p in range(8)]
    known = [(board, list(rng.multinomial(shots, truth.predicted_for(classical_winning_lines(board, player)))))
             for board, player in CALIBRATION_BOARDS]
    return readout_counts, known


def test_calibration_measures_the_double_win_factor():
    truth = NoiseProfile(0.4, IDENTITY, "truth", double_factor=0.7)
    from qttt.noise import CALIBRATION_BOARDS

    known = [(board, _expected_counts(truth, classical_winning_lines(board, player)))
             for board, player in CALIBRATION_BOARDS]
    fitted = NoiseProfile.from_calibration([[10**6 if m == p else 0 for m in range(8)] for p in range(8)], known)
    assert fitted.double_factor == pytest.approx(0.7, abs=1e-3)
    assert fitted.predicted_for({0, 7}) == pytest.approx(truth.predicted_for({0, 7}), abs=1e-4)


def test_bit_factors_stay_put_when_little_signal_survives():
    # 3% of the signal survives: at 1024 shots one board's fitted signal can be near 0. The old log fit let that
    # board drag its bits' factors anywhere (0.13 to the 2.0 cap on hardware); now a factor the shots cannot tell
    # from 1 stays 1, so repeated calibrations of the same device agree.
    truth = _profile(0.97)
    for seed in range(8):
        readout_counts, known = _calibration_counts(truth, 1024, seed)
        fitted = NoiseProfile.from_calibration(readout_counts, known)
        assert fitted.epsilon == pytest.approx(0.97, abs=0.015)
        for f, e in zip(fitted.bit_factors, fitted.signal_errors[1:], strict=True):
            assert f == 1.0 or abs(f - 1) >= 2 * e  # kept only when the data measures it


def test_no_signal_at_all_fits_one_epsilon():
    readout_counts, known = _calibration_counts(_profile(1.0), 1024, seed=3)
    fitted = NoiseProfile.from_calibration(readout_counts, known)
    assert fitted.bit_factors == (1.0, 1.0, 1.0)
    assert fitted.epsilon > 0.9
    assert fitted.signal_errors[1:] == (0.0, 0.0, 0.0)


def test_single_wins_measure_a_losing_lines_bias_too():
    # Line 5 gets 4% more than the model on every board where another line wins, but not on the no-win boards.
    from qttt.noise import CALIBRATION_BOARDS

    model = _profile(0.5)

    def biased(board, player):
        winners = classical_winning_lines(board, player)
        probs = list(model.predicted_for(winners))
        if winners and 5 not in winners:  # only where a line wins
            probs[5] *= 1.04
        return [round(p / sum(probs) * 10**5) for p in probs]

    readout_counts = [[10**6 if m == p else 0 for m in range(8)] for p in range(8)]
    all_boards = [(board, biased(board, player)) for board, player in CALIBRATION_BOARDS]
    no_win_only = [(b, c) for b, c in all_boards if not classical_winning_lines(b, X) | classical_winning_lines(b, O)]
    with_singles = NoiseProfile.from_calibration(readout_counts, all_boards)
    alone = NoiseProfile.from_calibration(readout_counts, no_win_only, epsilon=0.5)
    assert alone.no_win_shares[5] == pytest.approx(1 / 8, abs=1e-3)  # the no-win boards do not show it
    assert with_singles.no_win_shares[5] > 1 / 8 + 0.003
    assert with_singles.q0[5] > alone.q0[5]


def test_spread_between_boards_raises_the_floor():
    # Two no-win boards that disagree on line 7 far beyond shot noise: a board whose true share is above the
    # pooled mean must not be over-flagged, so the floor covers the spread, not only the mean's standard error.
    from qttt.noise import CALIBRATION_BOARDS

    readout_counts = [[10**6 if m == p else 0 for m in range(8)] for p in range(8)]
    no_win = ("X", "O", " ", " ", "O", "X", " ", "X", " ")
    single = (CALIBRATION_BOARDS[0][0], _expected_counts(_profile(0.3), {0}, shots=1000))
    agree = NoiseProfile.from_calibration(readout_counts, [single, (no_win, [125] * 8), (no_win, [125] * 8)])
    disagree = NoiseProfile.from_calibration(readout_counts, [single, (no_win, [100] * 7 + [300]),
                                                              (no_win, [125] * 8)])
    assert agree.no_win_spread[7] == pytest.approx(0.0, abs=1e-9)
    assert disagree.no_win_spread[7] > 0.05
    assert disagree.q0[7] > 0.125 + 2 * disagree.no_win_spread[7] * 0.9
    assert disagree.cutoffs(1024, DEFAULT_BUDGET)[7] > agree.cutoffs(1024, DEFAULT_BUDGET)[7]


def test_more_calibration_shots_shrink_the_margin():
    readout_counts, known = _calibration_counts(_profile(0.85), 1024, seed=1)
    fitted = NoiseProfile.from_calibration(readout_counts, known)
    more = fitted.with_calibration_shots(4096)
    assert more.no_win_errors == pytest.approx([e / 2 for e in fitted.no_win_errors])
    assert all(a <= b for a, b in zip(more.q0, fitted.q0, strict=True))
    # This profile needs more than its own 1024 calibration shots, so calibrating at the board shots helps.
    fixed = fitted.shots_needed(0.01)
    assert fixed > 1024
    assert fitted.shots_needed(0.01, in_job=True) < fixed


def test_empty_readout_row_is_refused():
    readout_counts = [[10**6 if m == p else 0 for m in range(8)] for p in range(8)]
    readout_counts[3] = [0] * 8
    with pytest.raises(ValueError, match="no shots"):
        NoiseProfile.from_calibration(readout_counts, epsilon=0.1)


def test_a_bit_no_calibration_board_has_keeps_factor_one():
    from qttt.noise import CALIBRATION_BOARDS

    known = [(CALIBRATION_BOARDS[0][0], _expected_counts(_profile(0.3), {0}))]  # line 0 only: every bit is 0
    fitted = NoiseProfile.from_calibration([[10**6 if m == p else 0 for m in range(8)] for p in range(8)], known)
    assert fitted.bit_factors == (1.0, 1.0, 1.0)
    assert fitted.epsilon == pytest.approx(0.3, abs=1e-4)


def test_measured_no_win_shares_raise_that_lines_cutoff():
    shares = (0.125,) * 3 + (0.16,) + (0.125,) * 4
    even = _profile(0.3)
    measured = NoiseProfile(0.3, IDENTITY, "test", no_win_shares=shares, no_win_shots=1024)
    assert measured.cutoffs(1024, 0.1)[3] > even.cutoffs(1024, 0.1)[3]
    assert measured.q0[3] == pytest.approx(0.16 + 2 * (0.16 * 0.84 / 1024) ** 0.5)
    assert measured.q0[0] >= even.q0[0]


def test_profiles_saved_before_bit_factors_still_load():
    from qttt.results import profile_summary

    saved = profile_summary(_profile(0.3))
    for key in ("bit_factors", "no_win_shares", "no_win_shots"):
        saved.pop(key)
    assert NoiseProfile.from_record(saved) == _profile(0.3)


def test_no_win_shares_are_pooled_over_the_no_win_boards():
    from qttt.noise import CALIBRATION_BOARDS

    readout_counts = [[10**6 if m == p else 0 for m in range(8)] for p in range(8)]
    no_win = ("X", "O", " ", " ", "O", "X", " ", "X", " ")
    high = [100, 100, 100, 100, 100, 100, 100, 300]  # line 7 high on one board only
    even = [125] * 8
    known = [(CALIBRATION_BOARDS[0][0], _expected_counts(_profile(0.3), {0}, shots=1000)), (no_win, high),
             (no_win, even)]
    fitted = NoiseProfile.from_calibration(readout_counts, known)
    assert fitted.no_win_shots == sum(sum(c) for _, c in known)
    # A weighted mean of the two no-win boards (the higher one is noisier), not the highest board.
    assert 0.125 < fitted.no_win_shares[7] < 0.3
