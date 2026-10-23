import pytest

from qttt.board import (
    EMPTY,
    LINE_NAMES,
    O,
    WIN_LINES,
    X,
    classical_winning_lines,
    legal_positions,
    other_player,
)

_ = EMPTY

# X on 1, 2, 3, 6, then 0: top row and left column at once, on turn 9.
DOUBLE_WIN = [X, X, X,
              X, O, O,
              X, O, O]


@pytest.fixture(scope="module")
def positions():
    return list(legal_positions())


def test_win_lines_are_eight_distinct_triples():
    assert len(WIN_LINES) == 8 == len(LINE_NAMES)
    assert len(set(WIN_LINES)) == 8
    for cells in WIN_LINES:
        assert len(set(cells)) == 3
        assert all(0 <= cell <= 8 for cell in cells)


def test_other_player():
    assert other_player(X) == O
    assert other_player(O) == X
    with pytest.raises(ValueError):
        other_player(EMPTY)


@pytest.mark.parametrize(
    ("board", "player", "expected"),
    [
        ([_] * 9, X, set()),
        ([X, X, X,
          O, O, _,
          _, _, _], X, {0}),
        ([O, X, X,
          X, O, _,
          _, _, O], O, {6}),
        ([X, O, X,
          _, O, _,
          X, O, _], O, {4}),
        ([X, O, X,
          X, O, O,
          O, X, X], X, set()),
    ],
)
def test_hand_picked_boards(board, player, expected):
    assert classical_winning_lines(board, player) == expected


def test_double_win_board():
    assert classical_winning_lines(DOUBLE_WIN, X) == {0, 3}
    assert classical_winning_lines(DOUBLE_WIN, O) == set()


def test_rejects_wrong_board_size():
    with pytest.raises(ValueError):
        classical_winning_lines([_] * 8, X)


def test_legal_position_counts(positions):
    terminal = sum(p.is_terminal for p in positions)
    print(f"{len(positions)} legal positions, {terminal} terminal")
    assert len(positions) == 5478
    assert len({p.board for p in positions}) == 5478
    assert terminal == 958


def test_last_player_matches_mark_counts(positions):
    for p in positions:
        xs, os = p.board.count(X), p.board.count(O)
        assert xs - os in (0, 1)
        if xs == os == 0:
            assert p.last_player is None
        else:
            assert p.last_player == (X if xs > os else O)


def test_only_last_player_can_have_a_line(positions):
    for p in positions:
        if p.last_player is None:
            continue
        assert classical_winning_lines(p.board, other_player(p.last_player)) == set()


def test_never_more_than_two_winning_lines(positions):
    # The adaptive detector relies on this: M is always 0, 1, or 2.
    for p in positions:
        if p.last_player is not None:
            assert len(classical_winning_lines(p.board, p.last_player)) <= 2


def test_double_wins_only_happen_on_turn_nine(positions):
    doubles = [
        p for p in positions
        if p.last_player is not None
        and len(classical_winning_lines(p.board, p.last_player)) == 2
    ]
    assert tuple(DOUBLE_WIN) in {p.board for p in doubles}
    for p in doubles:
        assert p.last_player == X
        assert EMPTY not in p.board
