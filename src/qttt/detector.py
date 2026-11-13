"""Winner detector: one Grover search, a check-budget cutoff, and classical checks of flagged lines."""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

from qiskit import QuantumCircuit

from qttt.backends import Backend, CircuitMetrics, IdealBackend
from qttt.board import EMPTY, WIN_LINES
from qttt.grover import build_search_circuit
from qttt.noise import DEFAULT_KNOWN, NoiseProfile, calibration_circuits, profile_from_calibration
from qttt.results import RunStore
from qttt.stats import NUM_LINES, _binomial_log_pmf, _binomial_tail, binomial_cutoff  # noqa: F401 (re-exported)

DEFAULT_SHOTS = 1024
DEFAULT_BUDGET = 0.1

@dataclass(frozen=True)
class SearchRun:
    """One run of the search circuit: counts per line index."""
    shots: int
    counts: tuple[int, ...]
    circuit: QuantumCircuit
    backend: str = "ideal"
    job_id: str | None = None
    transpiled: QuantumCircuit | None = None
    metrics: CircuitMetrics | None = None

    @property
    def probabilities(self) -> tuple[float, ...]:
        return tuple(c / self.shots for c in self.counts)

class Verdict(NamedTuple):
    """Flagged lines, the ones that failed the classical check, the confirmed lines and the winner."""
    candidates: tuple[int, ...]
    rejected: tuple[int, ...]
    lines: frozenset[int]
    winner: str | None

@dataclass(frozen=True)
class DetectionResult:
    """Detector verdict for one board, plus the run behind it."""
    board: tuple[str, ...]
    variant: str
    player: str | None
    winner: str | None
    lines: frozenset[int]
    candidates: tuple[int, ...]
    rejected: tuple[int, ...]
    cutoffs: tuple[int, ...]
    budget: float
    run: SearchRun
    profile: NoiseProfile | None = None

    @property
    def probabilities(self) -> tuple[float, ...]:
        return self.run.probabilities

    @property
    def checks(self) -> int:
        return len(self.candidates)

def run_searches(cases: Sequence[tuple[Sequence[str], str | None]], variant="player", shots=DEFAULT_SHOTS,
                 backend: Backend | None = None) -> list[SearchRun]:
    """Run the search for several (board, player) pairs as one job."""
    if backend is None:
        backend = IdealBackend()
    circuits, transpiled = _search_circuits(cases, variant, backend)
    out = backend.run(transpiled, shots)
    return _search_runs(out.counts, out.job_id, shots, circuits, transpiled, backend)

def _search_circuits(cases, variant, backend):
    circuits = [build_search_circuit(variant, board, player) for board, player in cases]
    transpiled = [backend.search_circuit(variant, board, player) for board, player in cases]
    return circuits, transpiled

def _search_runs(counts, job_id, shots, circuits, transpiled, backend):
    return [SearchRun(shots, c, qc, backend.name, job_id, tqc, backend.metrics(tqc))
            for c, qc, tqc in zip(counts, circuits, transpiled)]

def run_search(board, variant="player", player=None, shots=DEFAULT_SHOTS, backend=None) -> SearchRun:
    """Run the search circuit once and count how often each line index was measured."""
    return run_searches([(board, player)], variant, shots, backend)[0]

def flag_and_check(board, counts, cutoffs) -> Verdict:
    """Flag lines with count >= their cutoff and confirm each one on the board.

    cutoffs is one count for every line or 8 per-line counts. Only the cells of
    flagged lines are read.
    """
    if isinstance(cutoffs, int): cutoffs = (cutoffs,) * NUM_LINES
    candidates = tuple(i for i in range(NUM_LINES) if counts[i] >= cutoffs[i])
    confirmed = []
    rejected = []
    for i in candidates:
        a, b, c = WIN_LINES[i]
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            confirmed.append(i)
        else:
            rejected.append(i)
    lines = frozenset(confirmed)
    rejected = tuple(rejected)

    winner = board[WIN_LINES[min(lines)][0]] if lines else None
    return Verdict(candidates, rejected, lines, winner)

PROFILE_MODES = ("auto", "calibrate", "published", "ideal")

def detect_winners(cases: Sequence[tuple[Sequence[str], str | None]], variant="player", shots=DEFAULT_SHOTS,
                   budget=DEFAULT_BUDGET, backend: Backend | None = None,
                   profile: NoiseProfile | str = "auto",
                   store: RunStore | str | os.PathLike | None = None, tag: str | None = None) -> list[DetectionResult]:
    """detect_winner for several (board, player) pairs, run as one job.

    profile sets the per-line cutoffs:
      "auto" (default): "ideal" on the ideal simulator, "calibrate" on any other backend.
      "calibrate": add the 8 readout circuits and a known-winner board to the same
        job, fit a profile from them, and set the cutoffs from it once the job is back.
      "published": the backend's published calibration data (no extra circuits).
      "ideal": no noise, the same cutoff on every line.
      or a NoiseProfile measured earlier on the same physical qubits.
    Results are saved to store, labeled with tag; runs on real hardware are
    always saved, to results/ unless another store is given.
    """
    if backend is None:
        backend = IdealBackend()
    if isinstance(profile, str):
        if profile not in PROFILE_MODES:
            raise ValueError(f"Unknown profile {profile!r}; expected a NoiseProfile or one of {PROFILE_MODES}.")
        if profile == "auto":
            profile = "ideal" if isinstance(backend, IdealBackend) else "calibrate"
    NoiseProfile.from_ideal().cutoffs(shots, budget)  # check shots and budget before anything runs

    calibrating = profile == "calibrate"
    known = DEFAULT_KNOWN if variant == "player" else tuple((board, None) for board, _ in DEFAULT_KNOWN)
    circuits, transpiled = _search_circuits(cases, variant, backend)
    extra, line_qubits = calibration_circuits(backend, variant, known) if calibrating else ([], None)
    out = backend.run(extra + transpiled, shots)
    runs = _search_runs(out.counts[len(extra):], out.job_id, shots, circuits, transpiled, backend)

    if calibrating:
        profile = profile_from_calibration(backend, out.counts, line_qubits, known)
    elif profile == "published":
        profile = NoiseProfile.from_backend(backend, variant)
    elif profile == "ideal":
        profile = NoiseProfile.from_ideal()
    for run in runs:
        if profile.line_qubits is not None and run.metrics.line_qubits != profile.line_qubits:
            raise ValueError(f"profile was measured on line qubits {profile.line_qubits}, "
                             f"but this run reads out on {run.metrics.line_qubits}")
    cutoffs = profile.cutoffs(shots, budget)

    results = []
    for (board, player), run in zip(cases, runs):
        verdict = flag_and_check(board, run.counts, cutoffs)
        results.append(DetectionResult(
            board=tuple(board),
            variant=variant,
            player=player,
            winner=verdict.winner,
            lines=verdict.lines,
            candidates=verdict.candidates,
            rejected=verdict.rejected,
            cutoffs=cutoffs,
            budget=budget,
            run=run,
            profile=profile,
        ))

    if store is None and not backend.simulated:
        store = RunStore()
    if store is not None:
        if not isinstance(store, RunStore): store = RunStore(store)
        calibration_id = store.save_calibration(profile, out, backend, known, variant, tag) if calibrating else None
        for result in results:
            store.save_detection(result, backend, calibration_id, tag)
    return results

def detect_winner(board, variant="player", player=None, shots=DEFAULT_SHOTS, budget=DEFAULT_BUDGET,
                  backend=None, profile="auto", store=None) -> DetectionResult:
    """Detect whether anyone has won, using one Grover search plus classical checks.

    Lines measured at least their cutoff are checked on the board; any line
    reported in the result is a real winning line. The default 13-qubit
    "player" variant needs player, the one who just moved; the legacy 22-qubit
    "full" variant checks both players and ignores it. backend defaults to the
    ideal simulator; see detect_winners for profile and store.
    """
    return detect_winners([(board, player)], variant, shots, budget, backend, profile, store)[0]
