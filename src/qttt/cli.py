"""Play quantum tic-tac-toe in the terminal: `qttt` or `python -m qttt`.

Games are played on a simulator so every check answers in seconds. To see
how a real device judges the same boards, pass --verify ibm_kingston (or any
backend): when the games are over, every checked board goes to that device in
one job, calibration included. --backend ibm_kingston is shorthand for playing
on its snapshot, fake_kingston, and verifying on ibm_kingston.
"""

import argparse
import random
import warnings
from collections.abc import Callable

from qttt.backends import ANCILLA_FLAGS, DEFAULT_TRANSPILE_SEEDS, get_backend
from qttt.board import EMPTY, LINE_NAMES, O, X
from qttt.detector import DEFAULT_BUDGET, DEFAULT_SHOTS
from qttt.game import STRATEGIES, Game, TurnCheck, Verification, computer_move, verify_games
from qttt.grover import VARIANTS
from qttt.results import RunStore

BAR_WIDTH = 30


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qttt",
        description="Tic-tac-toe where a Grover search checks for the winner from turn 5 on.",
    )
    parser.add_argument("--backend", default="ideal",
                        help='where checks run during play: "ideal" (default) or a fake IBM device such as '
                             '"fake_kingston". A real device name plays on its fake snapshot and verifies on it.')
    parser.add_argument("--verify", default=None, metavar="BACKEND",
                        help="after the games, re-check every board on this backend in one job (e.g. ibm_kingston)")
    parser.add_argument("--games", type=int, default=1, help="games to play before verifying (default 1)")
    parser.add_argument("--variant", choices=VARIANTS, default="player",
                        help="12-qubit player oracle (default) or the legacy 22-qubit full oracle")
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS, help=f"shots per check (default {DEFAULT_SHOTS})")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET,
                        help=f"expected wasted checks per round (default {DEFAULT_BUDGET})")
    parser.add_argument("--seed", type=int, default=None, help="seed for the simulator and the computer player")
    parser.add_argument("--transpile-seeds", type=int, default=DEFAULT_TRANSPILE_SEEDS,
                        help=f"compiler seeds tried on IBM devices, best kept (default {DEFAULT_TRANSPILE_SEEDS})")
    parser.add_argument("--ancilla", choices=ANCILLA_FLAGS, default="auto",
                        help="extra qubit for a cheaper oracle: auto keeps the better compiled circuit (default)")
    parser.add_argument("--no-error-suppression", action="store_true",
                        help="real devices only: run without the default dynamical decoupling and gate twirling")
    parser.add_argument("--computer", choices=(X, O), default=None,
                        help="let the computer play X or O (default: two players at one keyboard)")
    parser.add_argument("--strategy", choices=STRATEGIES, default="rules",
                        help="computer strategy (default rules: win, block, center, corner)")
    parser.add_argument("--save", nargs="?", const="results", default=None, metavar="DIR",
                        help="save every check to DIR (default results/); real hardware runs are always saved")
    parser.add_argument("--yes", action="store_true", help="do not ask before sending a job to real hardware")
    return parser


def _backend(name: str, args):
    return get_backend(name, seed=args.seed, transpile_seeds=args.transpile_seeds, ancilla=ANCILLA_FLAGS[args.ancilla],
                       sampler_options={} if args.no_error_suppression else None)


def is_real_device(name: str) -> bool:
    return name != "ideal" and not name.startswith("fake_")


def fake_twin(name: str) -> str | None:
    """The fake snapshot of a real IBM device ("ibm_kingston" -> "fake_kingston"), if Qiskit has one."""
    from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

    twin = "fake_" + name.removeprefix("ibm_")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Properties of fake_nighthawk")
        names = {b.name for b in FakeProviderForBackendV2().backends()}
    return twin if twin in names else None


def main(argv: list[str] | None = None, *, read: Callable[[str], str] = input,
         write: Callable[[str], None] = print) -> int:
    args = build_parser().parse_args(argv)
    if args.shots < 1:
        write("--shots must be at least 1.")
        return 2
    if not 0 < args.budget <= 8:
        write("--budget must be in (0, 8].")
        return 2
    if args.transpile_seeds < 1:
        write("--transpile-seeds must be at least 1.")
        return 2
    if args.games < 1:
        write("--games must be at least 1.")
        return 2

    play_name, verify_name = args.backend, args.verify
    if is_real_device(play_name):
        if verify_name is None:
            verify_name = play_name
        play_name = fake_twin(play_name) or "ideal"
        write(f"Real hardware is too slow for live play: playing on {play_name}, "
              f"then verifying every board on {verify_name} in one job.")

    store = RunStore(args.save) if args.save else None
    play_backend = _backend(play_name, args)
    # Connect to the verify device now, so a missing account fails before the games, not after.
    verify_backend = _backend(verify_name, args) if verify_name is not None else None
    rng = random.Random(args.seed)
    write(f"Quantum tic-tac-toe: {args.variant} oracle on {play_backend.name}, {args.shots} shots per check.")
    write("Enter a cell number 0-8 to move, q to quit.\n")

    games = []
    for number in range(1, args.games + 1):
        if args.games > 1:
            write(f"Game {number} of {args.games}\n")
        game = Game(backend=play_backend, variant=args.variant, shots=args.shots, budget=args.budget, store=store)
        finished = play_game(game, args, rng, read, write)
        if game.checks:
            games.append(game)
        if not finished:
            break

    if verify_backend is not None:
        return verify(games, verify_name, verify_backend, args, store, read, write)
    return 0


def play_game(game: Game, args, rng: random.Random, read, write) -> bool:
    """Play one game to the end; False if the player quit."""
    while not game.is_over:
        write(render_board(game.board))
        player = game.current_player
        if player == args.computer:
            cell = computer_move(game.board, player, args.strategy, rng)
            write(f"Computer ({player}) plays {cell}.")
        else:
            cell = _ask_move(game, player, read, write)
            if cell is None:
                write("Quit.")
                return False
        check = game.play(cell)
        if check.result is not None:
            write(render_check(check))

    write(render_board(game.board))
    if game.winner:
        lines = ", ".join(LINE_NAMES[i] for i in sorted(game.winning_lines))
        write(f"{game.winner} wins ({lines}).")
    else:
        write("Draw.")
    if game.misses:
        write(f"The quantum check missed a win on {len(game.misses)} turn(s); the classical referee ended the game.")
    write("")
    return True


def verify(games: list[Game], name: str, backend, args, store, read, write) -> int:
    checks = sum(1 for g in games for c in g.checks if c.result is not None)
    if checks == 0:
        write("Nothing to verify: no game reached turn 5.")
        return 0
    if is_real_device(name) and not args.yes:
        folder = store.root if store else "results"
        write(f"Send {checks} board(s) and 9 calibration circuits to {name} as one job, {args.shots} shots each? "
              f"It waits in the provider's queue, uses your plan's time, and is saved to {folder}/.")
        try:
            answer = read("Continue? [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            write("Not sent.")
            return 1
    write(f"Verifying {checks} board(s) on {name} in one job...")
    verifications = verify_games(games, backend, shots=args.shots, budget=args.budget, store=store)
    write(render_verification(verifications, len(games) > 1))
    return 0


def _ask_move(game: Game, player: str, read, write) -> int | None:
    while True:
        try:
            text = read(f"{player} to move: ").strip().lower()
        except EOFError:  # input closed (Ctrl-D or end of a piped script): treat as quit
            return None
        if text in ("q", "quit", "exit"):
            return None
        if text.isdigit() and int(text) in game.legal_moves():
            return int(text)
        write(f"Pick an empty cell: {', '.join(map(str, game.legal_moves()))}.")


def render_board(board) -> str:
    """The board with empty cells shown by their number."""
    cells = [mark if mark != EMPTY else str(i) for i, mark in enumerate(board)]
    rows = [f" {cells[r]} | {cells[r + 1]} | {cells[r + 2]}" for r in (0, 3, 6)]
    return "\n---+---+---\n".join(rows) + "\n"


def render_check(check: TurnCheck) -> str:
    """The quantum check for one turn: a bar per line, the cutoffs, and the verdict."""
    result = check.result
    run = result.run
    peak = max(run.counts) or 1
    out = [f"Quantum check, turn {check.turn} ({check.player} just moved), {run.shots} shots:"]
    for i, count in enumerate(run.counts):
        bar = "#" * round(BAR_WIDTH * count / peak)
        if i in result.lines:
            note = "flagged, confirmed"
        elif i in result.rejected:
            note = "flagged, rejected"
        else:
            note = ""
        out.append(f"  {i} {LINE_NAMES[i]:36} {bar:{BAR_WIDTH}} {count:>5} / {result.cutoffs[i]:<5} {note}")
    if result.winner:
        out.append(f"  Quantum check: {result.winner} wins on {sorted(result.lines)}"
                   f" (classical checks: {result.checks}).")
    else:
        out.append(f"  Quantum check: no winner (classical checks: {result.checks}).")
    if check.missed:
        out.append(f"  MISSED: {check.player} has {sorted(check.referee_lines)}, but no line reached its cutoff. "
                   "The classical referee ends the game anyway.")
    elif check.missed_lines:
        out.append(f"  Partly missed: {sorted(check.missed_lines)} did not reach its cutoff; the win itself was found.")
    if result.profile is not None and result.profile.source != "ideal":
        out.append(f"  Noise profile: {result.profile.source}, ε = {result.profile.epsilon:.3f}")
    return "\n".join(out) + "\n"


def render_verification(verifications: list[Verification], show_game: bool) -> str:
    """One row per checked turn: what the referee says, what play showed, and what the device found."""
    first = verifications[0].result
    out = [f"Verified on {first.run.backend}"
           + (f" (job {first.run.job_id})" if first.run.job_id else "")
           + (f", noise profile {first.profile.source}, ε = {first.profile.epsilon:.3f}" if first.profile else "")
           + ":"]
    header = ("game " if show_game else "") + "turn player real lines   played   verified   rejected  verdict"
    out.append("  " + header)
    for v in verifications:
        verdict = "MISSED" if v.missed else "partly missed" if v.missed_lines else "correct"
        row = (f"{v.game + 1:<5}" if show_game else "") + (
            f"{v.check.turn:<4} {v.check.player:<6} {_lines(v.check.referee_lines):<12}"
            f" {_lines(v.check.quantum_lines):<8} {_lines(v.result.lines):<10} {_lines(v.result.rejected):<9}"
            f" {verdict}")
        out.append("  " + row)
    wins = [v for v in verifications if v.check.referee_lines]
    missed = sum(v.missed for v in verifications)
    false_flags = sum(len(v.result.rejected) for v in verifications)
    checks = sum(v.result.checks for v in verifications)
    out.append(f"  {len(verifications)} board(s), {len(wins) - missed} of {len(wins)} win(s) found, "
               f"{false_flags} false flag(s) rejected, {checks / len(verifications):.2f} classical checks per board.")
    return "\n".join(out) + "\n"


def _lines(lines) -> str:
    return ",".join(map(str, sorted(lines))) or "-"
