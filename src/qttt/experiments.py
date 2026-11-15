"""Collect labeled data: many random legal boards in one job, every result saved.

Boards are drawn (from a seed) from the boards the game checks, turn 5 on,
each with the player who just moved and its true label from the classical
checker: no win, single win or double win. collect() runs them through
detect_winners, so a noisy backend calibrates in the same job, and saves every
result with its label and a tag to the run store. summarize() scores the
results against the labels.
"""

import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from qttt.backends import Backend
from qttt.board import O, X, boards_by_win_count, classical_winning_lines
from qttt.detector import DEFAULT_BUDGET, DEFAULT_SHOTS, DetectionResult, detect_winners
from qttt.noise import NoiseProfile
from qttt.results import LABELS, RunStore

MIXES = ("balanced", "natural")


@dataclass(frozen=True)
class LabeledBoard:
    board: tuple[str, ...]
    player: str  # who just moved
    lines: frozenset[int]  # true winning lines

    @property
    def label(self) -> str:
        return LABELS[len(self.lines)]


def sample_boards(n: int | None = None, seed: int | None = None,
                  mix: str | Mapping[str, int] = "balanced") -> list[LabeledBoard]:
    """Draw n labeled boards.

    mix is "balanced" (as equal a split of no win / single win / double win as
    n allows), "natural" (in proportion to how often each occurs: 3,446 / 920 /
    22), or a mapping of label to count, e.g. {"single_win": 10, "double_win": 5},
    which sets n itself (pass n only if it matches the total).
    Within a label boards are distinct until that label runs out (there are
    only 22 double wins), then repeat. The order is shuffled.
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
        pool = groups[wins]
        k = counts.get(label, 0)
        chosen = rng.sample(pool, min(k, len(pool))) + [rng.choice(pool) for _ in range(k - len(pool))]
        for board in chosen:
            player = X if board.count(X) > board.count(O) else O  # who just moved
            picked.append(LabeledBoard(board, player, frozenset(classical_winning_lines(board, player))))
    rng.shuffle(picked)
    return picked


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
        tag = "collect-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
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
        rows = [f"{'label':11} {'boards':>6} {'win found':>10} {'all lines':>10} {'false flags':>11} {'checks/board':>12}"]
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
        base, extra = divmod(n, len(LABELS))
        return {label: base + (i < extra) for i, label in enumerate(LABELS)}
    if mix == "natural":
        sizes = [len(groups[w]) for w in range(len(LABELS))]
        total = sum(sizes)
        counts = [n * size // total for size in sizes]
        for i in sorted(range(len(sizes)), key=lambda i: n * sizes[i] / total - counts[i], reverse=True):
            if sum(counts) == n:
                break
            counts[i] += 1
        return dict(zip(LABELS, counts))
    raise ValueError(f"Unknown mix {mix!r}; expected one of {MIXES} or a mapping of label to count.")

