"""Collect labeled data: many random legal boards in one job, every result saved.

Boards are drawn (from a seed) from the boards the game checks, turn 5 on,
each with the player who just moved and its true label from the classical
checker: no win, single win or double win. collect() runs them through
detect_winners, so a noisy backend calibrates in the same job, and saves every
result with its label and a tag to the run store. summarize() scores the
results against the labels.
"""

import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from qttt.backends import Backend, JobOutput
from qttt.board import EMPTY, O, X, boards_by_win_count, classical_winning_lines
from qttt.detector import DEFAULT_BUDGET, DEFAULT_SHOTS, DetectionResult, detect_winners
from qttt.noise import NoiseProfile
from qttt.results import LABELS, RunStore
from qttt.stats import NUM_LINES

MIXES = ("balanced", "natural")
DOUBLE_REPEATS = 2  # how often "balanced" uses each of the 22 legal double wins


@dataclass(frozen=True)
class LabeledBoard:
    board: tuple[str, ...]
    player: str  # who just moved
    lines: frozenset[int]  # true winning lines

    @property
    def label(self) -> str:
        return LABELS[len(self.lines)]

    @property
    def turn(self) -> int:
        """Marks on the board (5-9), so also how many the player who just moved has."""
        return 9 - self.board.count(EMPTY)

    @property
    def stratum(self) -> tuple:
        """The group sample_boards balances within a label: (line, player) for a single win,
        (player, turn) for no win; double wins (all X on turn 9) form one group."""
        if len(self.lines) == 1:
            return (min(self.lines), self.player)
        if not self.lines:
            return (self.player, self.turn)
        return ()


def sample_boards(n: int | None = None, seed: int | None = None,
                  mix: str | Mapping[str, int] = "balanced") -> list[LabeledBoard]:
    """Draw n labeled boards.

    mix is "balanced", "natural" or a mapping of label to count:
      "balanced"  every legal double win DOUBLE_REPEATS times (there are only 22),
                  but at most a third of n, and the rest split evenly between
                  single wins and no-win boards (an odd one goes to single wins):
                  1000 boards are 44 double, 478 single and 478 no win.
      "natural"   in proportion to how often each occurs in real games (3,446 /
                  920 / 22), drawn at random within each label.
      a mapping   e.g. {"single_win": 10, "double_win": 5}, which sets n itself
                  (pass n only if it matches the total).

    With "balanced" or a mapping, boards within a label are also spread evenly
    over its strata (LabeledBoard.stratum): single wins cycle through the 8
    lines with alternating last mover, so every 16 cover each line once per
    player, no-win boards cycle through each (last mover, turn), and each draw
    takes the turn used least so far, so turns are spread too. Boards are
    distinct until a stratum runs out, then repeat in the same order, so every
    double win appears the same number of times (give or take one). The order
    is shuffled.
    """
    if isinstance(mix, Mapping):
        if n is not None and n != sum(mix.values()):
            raise ValueError(f"n is {n} but mix adds up to {sum(mix.values())}; pass one or make them match")
    elif n is None or n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    rng = random.Random(seed)
    groups = boards_by_win_count()
    counts = _label_counts(n, mix, groups)
    picked = []
    for wins, label in enumerate(LABELS):
        pool = [_labeled(board) for board in groups[wins]]
        k = counts.get(label, 0)
        if mix == "natural":
            picked += rng.sample(pool, min(k, len(pool))) + [rng.choice(pool) for _ in range(k - len(pool))]
        else:
            picked += _balanced(pool, k, rng)
    rng.shuffle(picked)
    return picked


def _labeled(board) -> LabeledBoard:
    player = X if board.count(X) > board.count(O) else O  # who just moved
    return LabeledBoard(board, player, frozenset(classical_winning_lines(board, player)))


def _balanced(pool: list[LabeledBoard], k: int, rng: random.Random) -> list[LabeledBoard]:
    """k boards drawn round-robin over the pool's strata; each draw takes the turn used least so far."""
    cells = defaultdict(lambda: defaultdict(list))  # stratum -> turn -> boards
    for b in pool:
        cells[b.stratum][b.turn].append(b)
    for turns in cells.values():
        for boards in turns.values():
            rng.shuffle(boards)
    order = _stratum_order(sorted(cells))
    used = Counter()  # (stratum, turn) -> boards taken
    turn_counts = Counter()
    chosen = []
    for i in range(k):
        key = order[i % len(order)]
        turns = cells[key]
        fresh = [t for t in turns if used[key, t] < len(turns[t])] or list(turns)  # all used: repeat
        turn = min(fresh, key=lambda t: (turn_counts[t], rng.random()))
        boards = turns[turn]
        chosen.append(boards[used[key, turn] % len(boards)])
        used[key, turn] += 1
        turn_counts[turn] += 1
    return chosen


def _stratum_order(keys):
    """Single-win strata as line 0-7 with X and O alternating (and swapped every second round), so
    any 8 in a row cover every line; other labels in sorted order with the players interleaved."""
    if keys and len(keys[0]) == 2 and isinstance(keys[0][0], int):  # (line, player)
        lines = sorted({line for line, _ in keys})
        order = [(line, (X, O)[(line + r) % 2]) for r in range(2) for line in lines]
        return [key for key in order if key in keys] + [key for key in keys if key not in order]
    return sorted(keys, key=lambda key: (key[1:], key[:1])) if keys and len(keys[0]) == 2 else list(keys)


DEFAULT_MAX_SHOTS = 4096  # the most shots per board choose_shots asks for, to protect a plan's device time
MISS_TARGET = 0.01  # choose_shots aims for at most this chance of missing a win


def choose_shots(backend: Backend, variant: str = "player", budget: float = DEFAULT_BUDGET,
                 max_shots: int = DEFAULT_MAX_SHOTS, store: RunStore | None = None,
                 tag: str | None = None) -> tuple[int, NoiseProfile]:
    """Shots per board for a batch, from a short calibration job on the backend itself.

    Runs only the calibration circuits (DEFAULT_SHOTS each) as a job of its own
    through detect_winners, so it is recorded the moment it is submitted and
    its counts the moment they return (resume_job finishes it if it is
    interrupted), and its calibration is saved to store (default results/).
    Returns the shots that calibration's profile predicts for at most
    MISS_TARGET misses within the false-flag budget, capped at max_shots, with
    the profile. When some line's winner cannot be told from a loser at any shot
    count up to the cap, the cap is returned; profile.miss_rates(shots, budget)
    shows what that costs.
    """
    if store is None:
        store = RunStore()
    detect_winners([], variant=variant, shots=DEFAULT_SHOTS, budget=budget, backend=backend,
                   profile="calibrate", store=store, tag=tag)
    submitted = [r for r in store.load("job") if r["status"] == "submitted"][-1]
    (record,) = [r for r in store.load("calibration") if r.get("job_record") == submitted["id"]]
    profile = replace(NoiseProfile.from_record(record["profile"]), calibration_id=record["id"])
    return profile.shots_needed(MISS_TARGET, budget, max_shots) or max_shots, profile


def default_tag() -> str:
    """A batch name from the current time, e.g. collect-20261115T093000Z."""
    return "collect-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def saved_profile(store: RunStore, tags: Sequence[str], backend: Backend,
                  variant: str = "player") -> tuple[NoiseProfile, list[dict]]:
    """One profile from calibrations saved earlier (found by tag), so a batch can skip its own calibration.

    Each calibration must have run on this backend with the same Sampler
    options and variant, and on the line qubits this backend's search circuit
    reads out on now (a device can move them when it is recalibrated). Their
    readout counts and known boards are pooled and fitted as one calibration.
    Returns the profile and the records; save_pooled_calibration records it.
    """
    records = [r for r in store.load("calibration") if r["tag"] in set(tags)]
    missing = set(tags) - {r["tag"] for r in records}
    if missing:
        raise ValueError(f"no calibration saved with tag {', '.join(sorted(missing))}")
    line_qubits = backend.metrics(backend.search_template(variant)).line_qubits
    options = backend.options().get("sampler_options")
    for r in records:
        problems = []
        if r["backend"]["name"] != backend.name:
            problems.append(f"it ran on {r['backend']['name']}")
        if r["backend"]["options"].get("sampler_options") != options:
            problems.append(f"it ran with Sampler options {r['backend']['options'].get('sampler_options') or 'none'},"
                            f" this run uses {options or 'none'}")
        if r["variant"] != variant:
            problems.append(f"it is for the {r['variant']} variant")
        if tuple(r["profile"]["line_qubits"]) != tuple(line_qubits):
            problems.append(f"it measured line qubits {r['profile']['line_qubits']}, but the circuit now reads out"
                            f" on {list(line_qubits)}")
        if problems:
            raise ValueError(f"calibration {r['tag']} does not fit this run: {'; '.join(problems)}")
    readout = [[sum(r["readout_counts"][p][m] for r in records) for m in range(NUM_LINES)] for p in range(NUM_LINES)]
    known = [(tuple(k["board"]), k["counts"]) for r in records for k in r["known"]]
    profile = NoiseProfile.from_calibration(readout, known, source=f"calibration:{backend.name}",
                                            line_qubits=tuple(line_qubits))
    return profile, records


def save_pooled_calibration(store: RunStore, profile: NoiseProfile, records: list[dict], backend: Backend,
                            variant: str = "player", tag: str | None = None) -> NoiseProfile:
    """Save a saved_profile() result as its own calibration record; returns the profile linked to it."""
    readout = [[sum(r["readout_counts"][p][m] for r in records) for m in range(NUM_LINES)] for p in range(NUM_LINES)]
    known = [(k["board"], k["player"]) for r in records for k in r["known"]]
    counts = [tuple(k["counts"]) for r in records for k in r["known"]]
    output = JobOutput(tuple(tuple(row) for row in readout) + tuple(counts), None)
    calibration_id = store.save_calibration(profile, output, backend, known, variant, tag,
                                            pooled_from=[r["id"] for r in records])
    return replace(profile, calibration_id=calibration_id)


def collect(boards: Sequence[LabeledBoard], backend: Backend, variant: str = "player", shots: int = DEFAULT_SHOTS,
            budget: float = DEFAULT_BUDGET, profile: NoiseProfile | str = "auto",
            store: RunStore | None = None, tag: str | None = None,
            boards_per_job: int | None = None) -> list[DetectionResult]:
    """Run every board and save each result with its label.

    One job holds the calibration circuits (on a noisy backend) and up to
    boards_per_job boards (default: all of them). Results are saved to store
    (default results/) under tag (default: a timestamp).
    """
    if boards_per_job is not None and boards_per_job < 1:
        raise ValueError(f"boards_per_job must be at least 1, got {boards_per_job}")
    if not boards:
        return []
    if store is None:
        store = RunStore()
    if tag is None:
        tag = default_tag()
    size = boards_per_job or len(boards)
    results = []
    for start in range(0, len(boards), size):
        chunk = boards[start:start + size]
        cases = [(b.board, b.player if variant == "player" else None) for b in chunk]
        results += detect_winners(cases, variant=variant, shots=shots, budget=budget, backend=backend,
                                  profile=profile, store=store, tag=tag)
    return results


@dataclass(frozen=True)
class Summary:
    """How the results compare with the true labels, per label."""
    boards: Counter
    wins_found: Counter  # boards with a win where at least one line was confirmed
    all_lines_found: Counter  # boards where every winning line was confirmed
    false_flags: Counter  # rejected checks
    checks: Counter  # classical checks used

    def text(self) -> str:
        rows = [f"{'label':11} {'boards':>6} {'win found':>10} {'all lines':>10} {'false flags':>11}"
                f" {'checks/board':>12}"]
        for label in LABELS:
            n = self.boards[label]
            if not n:
                continue
            found = f"{self.wins_found[label]}/{n}" if label != "no_win" else "-"
            every = f"{self.all_lines_found[label]}/{n}" if label != "no_win" else "-"
            rows.append(f"{label:11} {n:>6} {found:>10} {every:>10} {self.false_flags[label]:>11}"
                        f" {self.checks[label] / n:>12.2f}")
        return "\n".join(rows)


def summarize(boards: Sequence[LabeledBoard], results: Sequence[DetectionResult]) -> Summary:
    """Score results against the labels they were collected with (same order)."""
    s = Summary(Counter(), Counter(), Counter(), Counter(), Counter())
    for b, r in zip(boards, results, strict=True):
        s.boards[b.label] += 1
        s.wins_found[b.label] += bool(r.lines)
        s.all_lines_found[b.label] += r.lines == b.lines
        s.false_flags[b.label] += len(r.rejected)
        s.checks[b.label] += r.checks
    return s


def _label_counts(n, mix, groups):
    if isinstance(mix, Mapping):
        unknown = set(mix) - set(LABELS)
        if unknown:
            raise ValueError(f"Unknown labels {sorted(unknown)}; expected {LABELS}.")
        return dict(mix)
    if mix == "balanced":
        double = min(DOUBLE_REPEATS * len(groups[2]), n // 3)
        single = -(-(n - double) // 2)
        return {"no_win": n - double - single, "single_win": single, "double_win": double}
    if mix == "natural":
        sizes = [len(groups[w]) for w in range(len(LABELS))]
        total = sum(sizes)
        counts = [n * size // total for size in sizes]
        for i in sorted(range(len(sizes)), key=lambda i: n * sizes[i] / total - counts[i], reverse=True):
            if sum(counts) == n:
                break
            counts[i] += 1
        return dict(zip(LABELS, counts, strict=True))
    raise ValueError(f"Unknown mix {mix!r}; expected one of {MIXES} or a mapping of label to count.")

