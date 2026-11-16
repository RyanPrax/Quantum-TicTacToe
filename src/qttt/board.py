"""Classical tic-tac-toe board: winning lines, winner check, and legal positions.

A board is a 9-item sequence of "X", "O", or " " (empty), indexed:

    0 | 1 | 2
    3 | 4 | 5
    6 | 7 | 8
"""

from collections.abc import Iterator, Sequence
from typing import NamedTuple

EMPTY = " "
X = "X"
O = "O"
PLAYERS = (X, O)

# Each line is stored as cell indices. Its position in this list (0-7) is the
# line index the Grover search measures.
WIN_LINES: list[tuple[int, int, int]] = [
    (0, 1, 2),  # 0: top row
    (3, 4, 5),  # 1: middle row
    (6, 7, 8),  # 2: bottom row
    (0, 3, 6),  # 3: left column
    (1, 4, 7),  # 4: middle column
    (2, 5, 8),  # 5: right column
    (0, 4, 8),  # 6: diagonal (top-left to bottom-right)
    (2, 4, 6),  # 7: diagonal (top-right to bottom-left)
]

LINE_NAMES: list[str] = [
    "top row",
    "middle row",
    "bottom row",
    "left column",
    "middle column",
    "right column",
    "diagonal (top-left to bottom-right)",
    "diagonal (top-right to bottom-left)",
]

# Shorter names for tables and figure titles.
SHORT_LINE_NAMES: tuple[str, ...] = ("top row", "middle row", "bottom row", "left column", "middle column",
                                     "right column", "diagonal from top-left", "diagonal from top-right")


class Position(NamedTuple):
    """A board reachable by actual play."""

    board: tuple[str, ...]
    last_player: str | None  # None for the empty board
    is_terminal: bool  # someone won, or the board is full


def other_player(player: str) -> str:
    """Return the opponent of the given player."""
    if player == X:
        return O
    if player == O:
        return X
    raise ValueError(f"Unknown player: {player!r}")


def classical_winning_lines(board: Sequence[str], player: str) -> set[int]:
    """Return the indices of every line the player has filled.

    This is the reference that every quantum oracle is tested against.
    """
    if len(board) != 9:
        raise ValueError("Board must have exactly 9 cells.")
    return {
        index
        for index, cells in enumerate(WIN_LINES)
        if all(board[cell] == player for cell in cells)
    }


def legal_positions() -> Iterator[Position]:
    """Yield every distinct board reachable by actual play, including the empty board.

    Walks the game tree from the empty board with X moving first, and stops
    expanding a board once someone has won or it is full. There are 5,478
    positions in total, 958 of them terminal.
    """
    start = (EMPTY,) * 9
    seen = {start}
    stack: list[tuple[tuple[str, ...], str | None]] = [(start, None)]
    while stack:
        board, last_player = stack.pop()
        won = last_player is not None and bool(
            classical_winning_lines(board, last_player)
        )
        is_terminal = won or EMPTY not in board
        yield Position(board, last_player, is_terminal)
        if is_terminal:
            continue

        mover = X if last_player is None else other_player(last_player)
        for cell, mark in enumerate(board):
            if mark != EMPTY:
                continue
            child = board[:cell] + (mover,) + board[cell + 1 :]
            if child not in seen:
                seen.add(child)
                stack.append((child, mover))


def boards_by_win_count() -> dict[int, list[tuple[str, ...]]]:
    """Group the boards the game checks for a winner by how many lines the last player filled.

    The game checks from turn 5 onward, the first turn a line is possible.
    Keys are 0, 1 and 2 (3,446, 920 and 22 boards); each list is sorted.
    """
    groups: dict[int, list[tuple[str, ...]]] = {0: [], 1: [], 2: []}
    for p in legal_positions():
        if p.last_player is None or 9 - p.board.count(EMPTY) < 5:
            continue
        groups[len(classical_winning_lines(p.board, p.last_player))].append(p.board)
    for boards in groups.values():
        boards.sort()
    return groups
