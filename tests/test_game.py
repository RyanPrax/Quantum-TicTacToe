"""The game loop, the classical referee, the computer player and verify_games."""

import random

import pytest

from qttt.backends import IdealBackend
from qttt.board import EMPTY, O, X
from qttt.game import FIRST_CHECK_TURN, Game, computer_move, verify_games
from qttt.noise import CALIBRATION_CIRCUITS

SEED = 4903


class _NoSignalBackend(IdealBackend):
    """Ideal backend whose counts carry no Grover signal, as if noise had erased it."""

    def run(self, circuits, shots, on_submit=None):
        out = super().run(circuits, shots, on_submit)
        flat = tuple(shots // 8 for _ in range(8))
        return type(out)(tuple(flat for _ in out.counts), out.job_id)


class _CountingBackend(IdealBackend):
    def __init__(self, seed=None):
        super().__init__(seed)
        self.jobs = []

    def run(self, circuits, shots, on_submit=None):
        self.jobs.append(len(circuits))
        return super().run(circuits, shots, on_submit)


def _play(game, cells):
    return [game.play(c) for c in cells]


def test_turn_order_and_legal_moves():
    game = Game(backend=IdealBackend(seed=SEED))
    assert game.current_player == X
    game.play(4)
    assert game.current_player == O
    assert game.board[4] == X
    assert 4 not in game.legal_moves()


@pytest.mark.parametrize("cell", [-1, 9, "4", 4.0])
def test_bad_cells_are_refused(cell):
    with pytest.raises(ValueError):
        Game(backend=IdealBackend(seed=SEED)).play(cell)


def test_taken_cell_is_refused():
    game = Game(backend=IdealBackend(seed=SEED))
    game.play(0)
    with pytest.raises(ValueError, match="taken"):
        game.play(0)


def test_no_quantum_check_before_turn_five():
    game = Game(backend=IdealBackend(seed=SEED))
    checks = _play(game, [0, 3, 1, 4])
    assert all(c.result is None for c in checks)
    assert FIRST_CHECK_TURN == 5


def test_single_win_ends_the_game():
    game = Game(backend=IdealBackend(seed=SEED), shots=512)
    checks = _play(game, [0, 3, 1, 4, 2])  # X takes the top row on turn 5
    last = checks[-1]
    assert last.result.winner == X and last.result.lines == {0}
    assert not last.missed
    assert game.is_over and game.winner == X and game.winning_lines == {0}
    with pytest.raises(ValueError, match="over"):
        game.play(8)


def test_double_win_on_turn_nine():
    game = Game(backend=IdealBackend(seed=SEED), shots=512)
    _play(game, [1, 4, 2, 5, 3, 7, 6, 8, 0])  # ends X X X / X O O / X O O
    assert game.winner == X
    assert game.winning_lines == {0, 3}
    assert game.checks[-1].result.lines == {0, 3}


def test_draw():
    game = Game(backend=IdealBackend(seed=SEED), shots=512)
    _play(game, [0, 1, 2, 4, 3, 5, 7, 6, 8])  # X O X / X O O / O X X
    assert game.is_draw and game.winner is None
    assert all(c.result.winner is None for c in game.checks if c.result)


def test_missed_win_still_ends_the_game_and_is_recorded():
    game = Game(backend=_NoSignalBackend(seed=SEED), shots=512)
    checks = _play(game, [0, 3, 1, 4, 2])
    last = checks[-1]
    assert last.missed
    assert last.missed_lines == {0}
    assert game.winner == X  # the classical referee ends it anyway
    assert game.misses == [last]


def test_calibration_runs_once_per_game():
    backend = _CountingBackend(seed=SEED)
    game = Game(backend=backend, shots=256, profile="calibrate")
    _play(game, [0, 1, 2, 4, 3, 5, 7, 6, 8])  # a draw: 5 checks
    assert backend.jobs == [CALIBRATION_CIRCUITS + 1, 1, 1, 1, 1]  # calibration only in the first job
    profiles = {c.result.profile for c in game.checks if c.result}
    assert len(profiles) == 1


def test_board_snapshot_is_taken_after_the_move():
    game = Game(backend=IdealBackend(seed=SEED))
    checks = _play(game, [0, 3, 1, 4, 2])
    assert checks[-1].board == (X, X, X, O, O, EMPTY, EMPTY, EMPTY, EMPTY)
    assert checks[0].board[0] == X and checks[0].board.count(EMPTY) == 8


# --- computer player ---

def test_rules_strategy_wins_when_it_can():
    board = [X, X, EMPTY, O, O, EMPTY, EMPTY, EMPTY, EMPTY]
    assert computer_move(board, X, "rules") == 2
    assert computer_move(board, O, "rules") == 5


def test_rules_strategy_blocks():
    board = [X, X, EMPTY, EMPTY, O, EMPTY, EMPTY, EMPTY, EMPTY]
    assert computer_move(board, O, "rules") == 2


def test_rules_strategy_prefers_center_then_corners():
    assert computer_move([EMPTY] * 9, X, "rules") == 4
    board = [EMPTY] * 9
    board[4] = X
    assert computer_move(board, O, "rules", random.Random(1)) in (0, 2, 6, 8)


def test_random_strategy_picks_an_empty_cell():
    board = [X, O, X, O, X, O, EMPTY, X, O]
    assert computer_move(board, O, "random", random.Random(1)) == 6


def test_computer_move_rejects_bad_input():
    with pytest.raises(ValueError):
        computer_move([EMPTY] * 9, X, "clever")
    with pytest.raises(ValueError):
        computer_move([X] * 9, X)


def test_two_computers_always_finish_a_game():
    rng = random.Random(SEED)
    game = Game(backend=IdealBackend(seed=SEED), shots=256)
    while not game.is_over:
        game.play(computer_move(game.board, game.current_player, "random", rng))
    assert game.is_over


# --- verify_games ---

def test_verify_games_sends_every_check_in_one_job():
    games = []
    for cells in ([0, 3, 1, 4, 2], [0, 1, 2, 4, 3, 5, 7, 6, 8]):
        game = Game(backend=IdealBackend(seed=SEED), shots=256)
        _play(game, cells)
        games.append(game)
    backend = _CountingBackend(seed=SEED)
    verifications = verify_games(games, backend, shots=256)
    assert len(verifications) == 1 + 5
    assert backend.jobs == [1 + 5]  # ideal backend: no calibration circuits
    assert [v.game for v in verifications] == [0, 1, 1, 1, 1, 1]
    assert not any(v.missed for v in verifications)
    assert verifications[0].result.winner == X


def test_verify_games_reports_misses():
    game = Game(backend=IdealBackend(seed=SEED), shots=256)
    _play(game, [0, 3, 1, 4, 2])
    (verification,) = verify_games([game], _NoSignalBackend(seed=SEED), shots=256)
    assert verification.missed
    assert not verification.check.missed  # it was found during play


def test_verify_games_with_nothing_to_check():
    game = Game(backend=IdealBackend(seed=SEED))
    _play(game, [0, 1])
    assert verify_games([game], IdealBackend()) == []


def test_verify_games_needs_one_variant():
    a = Game(backend=IdealBackend(seed=SEED), shots=64)
    _play(a, [0, 3, 1, 4, 2])
    b = Game(backend=IdealBackend(seed=SEED), shots=64, variant="full")
    b.board = [X, X, EMPTY, O, O, EMPTY, EMPTY, EMPTY, EMPTY]  # skip ahead without 22-qubit checks
    b.play(2)
    with pytest.raises(ValueError, match="same variant"):
        verify_games([a, b], IdealBackend())


def test_every_turn_links_to_the_games_calibration(tmp_path):
    from qttt.results import RunStore

    store = RunStore(tmp_path)
    game = Game(backend=IdealBackend(seed=SEED), shots=256, profile="calibrate", store=store)
    _play(game, [0, 1, 2, 4, 3, 5, 7])  # no winner yet; checks on turns 5, 6 and 7
    (calibration,) = store.load("calibration")  # calibrated once, on turn 5
    detections = store.load("detection")
    assert len(detections) == 3
    assert {d["calibration_id"] for d in detections} == {calibration["id"]}
