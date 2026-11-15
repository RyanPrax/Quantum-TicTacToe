"""A game of tic-tac-toe that checks for a winner with the Grover detector.

From turn 5 (the first turn a line is possible) every move is followed by a
quantum check of the board for the player who just moved. A classical referee
checks the same board, so the game always ends correctly: if noise makes the
quantum check miss a win, the game still ends, and the miss is recorded on that
turn's check so it can be reported.

On a noisy backend the noise profile is calibrated with the first check and
reused for the rest of the game, so later turns run only their own circuit.

Real hardware is too slow for live play (every check would wait in the
provider's queue), so games are played on a simulator and verify_games() sends
every checked board of one or more games to the device afterwards, in one job.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass

from qttt.backends import Backend, IdealBackend
from qttt.board import EMPTY, WIN_LINES, X, classical_winning_lines, other_player
from qttt.detector import DEFAULT_BUDGET, DEFAULT_SHOTS, DetectionResult, detect_winner, detect_winners
from qttt.noise import NoiseProfile
from qttt.results import RunStore

FIRST_CHECK_TURN = 5  # a line needs 3 marks from one player, so 5 marks in total
STRATEGIES = ("random", "rules")


@dataclass(frozen=True)
class TurnCheck:
    """The winner check after one move: the quantum result and the classical referee's answer."""
    turn: int
    player: str
    cell: int
    board: tuple[str, ...]  # the board after the move
    result: DetectionResult | None  # None before FIRST_CHECK_TURN, when no line is possible
    referee_lines: frozenset[int]  # lines the player really has

    @property
    def quantum_lines(self) -> frozenset[int]:
        return self.result.lines if self.result is not None else frozenset()

    @property
    def missed(self) -> bool:
        """The player won but the quantum check confirmed no line."""
        return bool(self.referee_lines) and not self.quantum_lines

    @property
    def missed_lines(self) -> frozenset[int]:
        """Winning lines the quantum check did not confirm (one of two counts as a partial miss)."""
        return self.referee_lines - self.quantum_lines


class Game:
    """Board, turn order and a winner check after every move."""

    def __init__(self, backend: Backend | None = None, variant: str = "player", shots: int = DEFAULT_SHOTS,
                 budget: float = DEFAULT_BUDGET, profile: NoiseProfile | str = "auto",
                 store: RunStore | None = None):
        self.board = [EMPTY] * 9
        self.backend = backend if backend is not None else IdealBackend()
        self.variant = variant
        self.shots = shots
        self.budget = budget
        self.profile = profile
        self.store = store
        self.checks: list[TurnCheck] = []
        self.winner: str | None = None
        self.winning_lines: frozenset[int] = frozenset()

    @property
    def turn(self) -> int:
        """Number of marks placed so far."""
        return 9 - self.board.count(EMPTY)

    @property
    def current_player(self) -> str:
        return X if self.turn % 2 == 0 else other_player(X)

    @property
    def is_over(self) -> bool:
        return self.winner is not None or EMPTY not in self.board

    @property
    def is_draw(self) -> bool:
        return self.is_over and self.winner is None

    @property
    def misses(self) -> list[TurnCheck]:
        """Checks where the quantum detector missed a win, fully or in part."""
        return [c for c in self.checks if c.missed_lines]

    def legal_moves(self) -> list[int]:
        return [] if self.is_over else [c for c in range(9) if self.board[c] == EMPTY]

    def play(self, cell: int) -> TurnCheck:
        """Place the current player's mark, check for a winner, and return the check."""
        if self.is_over:
            raise ValueError("The game is over.")
        if not isinstance(cell, int) or not 0 <= cell <= 8:
            raise ValueError(f"Cell must be 0-8, got {cell!r}.")
        if self.board[cell] != EMPTY:
            raise ValueError(f"Cell {cell} is taken.")

        player = self.current_player
        self.board[cell] = player
        result = None
        if self.turn >= FIRST_CHECK_TURN:
            result = detect_winner(self.board, variant=self.variant, player=player, shots=self.shots,
                                   budget=self.budget, backend=self.backend, profile=self.profile,
                                   store=self.store)
            if isinstance(self.profile, str) and result.profile.source.startswith("calibration:"):
                self.profile = result.profile  # calibrated once; reuse it for the rest of the game
        referee = frozenset(classical_winning_lines(self.board, player))
        check = TurnCheck(self.turn, player, cell, tuple(self.board), result, referee)
        self.checks.append(check)
        if referee:
            self.winner = player
            self.winning_lines = referee
        return check


@dataclass(frozen=True)
class Verification:
    """One played turn re-checked on another backend (usually real hardware)."""
    game: int  # index into the games passed to verify_games
    check: TurnCheck  # the turn as played
    result: DetectionResult  # the check on the verification backend

    @property
    def missed(self) -> bool:
        """The player won but this backend confirmed no line."""
        return bool(self.check.referee_lines) and not self.result.lines

    @property
    def missed_lines(self) -> frozenset[int]:
        return self.check.referee_lines - self.result.lines


def verify_games(games: Sequence[Game], backend: Backend, shots: int = DEFAULT_SHOTS,
                 budget: float = DEFAULT_BUDGET, store: RunStore | None = None) -> list[Verification]:
    """Re-run every quantum check of the games on backend, all in one job.

    The job also holds the calibration circuits (detect_winners' default on a
    noisy backend), so it has one queue wait and the cutoffs come from the
    device's noise at that moment. Runs on real hardware are always saved.
    """
    entries = [(i, check) for i, game in enumerate(games) for check in game.checks if check.result is not None]
    if not entries:
        return []
    variants = {games[i].variant for i, _ in entries}
    if len(variants) != 1:
        raise ValueError(f"All games must use the same variant to share one job, got {sorted(variants)}.")
    results = detect_winners([(check.board, check.player) for _, check in entries], variant=variants.pop(),
                             shots=shots, budget=budget, backend=backend, store=store)
    return [Verification(i, check, result) for (i, check), result in zip(entries, results, strict=True)]


def computer_move(board: Sequence[str], player: str, strategy: str = "rules",
                  rng: random.Random | None = None) -> int:
    """Pick a move for the computer.

    "random" picks any empty cell. "rules" wins if it can, blocks the
    opponent's win, then prefers the center, then a corner, then any cell.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; expected one of {STRATEGIES}.")
    rng = rng or random.Random()
    empty = [c for c in range(9) if board[c] == EMPTY]
    if not empty:
        raise ValueError("No empty cells.")
    if strategy == "random":
        return rng.choice(empty)
    for who in (player, other_player(player)):  # win, then block
        for cell in empty:
            if _completes_line(board, cell, who):
                return cell
    if 4 in empty:
        return 4
    corners = [c for c in (0, 2, 6, 8) if c in empty]
    return rng.choice(corners or empty)


def _completes_line(board, cell, player):
    return any(cell in line and all(board[c] == player for c in line if c != cell) for line in WIN_LINES)
