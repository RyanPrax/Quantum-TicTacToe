"""Winner detector: one Grover search, a check-budget cutoff, and classical checks of flagged lines."""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

from qiskit import QuantumCircuit

from qttt.backends import Backend, CircuitMetrics, IdealBackend, JobOutput, line_qubits
from qttt.board import EMPTY, WIN_LINES
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
    circuits = [backend.logical_search_circuit(variant, board, player) for board, player in cases]
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
    A profile that does not fit the circuits is refused before anything runs.
    Results are saved to store, labeled with tag; runs on real hardware are
    always saved, to results/ unless another store is given. The store records
    the job when it is submitted and its counts when they are back, so a job
    interrupted after that can be finished with resume_job.
    """
    if backend is None:
        backend = IdealBackend()
    if isinstance(profile, str):
        if profile not in PROFILE_MODES:
            raise ValueError(f"Unknown profile {profile!r}; expected a NoiseProfile or one of {PROFILE_MODES}.")
        if profile == "auto":
            profile = "ideal" if isinstance(backend, IdealBackend) else "calibrate"
    NoiseProfile.from_ideal().cutoffs(shots, budget)  # check shots and budget before anything runs
    store = _open_store(store, backend)

    calibrating = profile == "calibrate"
    known = _known(variant) if calibrating else ()
    circuits, transpiled = _search_circuits(cases, variant, backend)
    extra, line_qubits = calibration_circuits(backend, variant, known) if calibrating else ([], None)
    if not calibrating:
        profile = _fixed_profile(profile, backend, variant)
        _check_line_qubits(profile, transpiled)

    submitted = []
    def on_submit(job_id):
        if store is not None:
            submitted.append(store.save_job_submitted(
                job_id, backend, backend.search_template(variant), cases=cases, variant=variant, shots=shots,
                budget=budget, known=known, profile=None if calibrating else profile, tag=tag))
    out = backend.run(extra + transpiled, shots, on_submit)
    job_record = submitted[0] if submitted else None
    try:
        if store is not None:
            store.save_job_output(job_record, out)
        return _finish(cases, variant, shots, budget, backend, out, circuits, transpiled, len(extra),
                       None if calibrating else profile, known, line_qubits, store, tag, job_record)
    except Exception as e:
        if job_record is not None:
            e.add_note(f"Job {out.job_id or job_record} ran and is recorded in {store.runs_file}; "
                       f"finish it with resume_job({job_record!r}, backend, store).")
        raise

def resume_job(job: str, backend: Backend, store: RunStore | str | os.PathLike = "results") -> list[DetectionResult]:
    """Finish a job that detect_winners submitted but did not finish (lost kernel, crash, Ctrl-C).

    job is the provider's job id or the id of the job's submitted record. The
    counts are taken from the store if they were saved, or else fetched from the
    provider by job id. The circuits are rebuilt from the template saved with the
    job, so the cutoffs and records come out as if the run had not stopped.
    """
    if not isinstance(store, RunStore): store = RunStore(store)
    submitted, done = store.job(job)
    if submitted["backend"]["name"] != backend.name:
        raise ValueError(f"job ran on {submitted['backend']['name']}, not {backend.name}")
    if any(r.get("job_record") == submitted["id"] for r in store.load("detection")):
        raise ValueError(f"job {job} is already finished; its detections are in {store.runs_file}")

    variant = submitted["variant"]
    cases = [(tuple(c["board"]), c["player"]) for c in submitted["cases"]]
    known = tuple((tuple(k["board"]), k["player"]) for k in submitted["known"])
    backend.set_search_template(variant, store.load_circuit(submitted))
    circuits, transpiled = _search_circuits(cases, variant, backend)
    extra, line_qubits = calibration_circuits(backend, variant, known) if submitted["calibrate"] else ([], None)
    if done is not None:
        out = JobOutput(tuple(tuple(c) for c in done["counts"]), done["job_id"])
    else:
        out = backend.fetch(submitted["job_id"])
        store.save_job_output(submitted["id"], out)
    saved = submitted["profile"]
    profile = None if saved is None else NoiseProfile(
        epsilon=saved["epsilon"], readout=tuple(tuple(row) for row in saved["readout"]), source=saved["source"],
        line_qubits=None if saved["line_qubits"] is None else tuple(saved["line_qubits"]))
    return _finish(cases, variant, submitted["shots"], submitted["budget"], backend, out, circuits, transpiled,
                   len(extra), profile, known, line_qubits, store, submitted["tag"], submitted["id"])

def _known(variant):
    return DEFAULT_KNOWN if variant == "player" else tuple((board, None) for board, _ in DEFAULT_KNOWN)

def _open_store(store, backend):
    if store is None and not backend.simulated:
        store = RunStore()
    if store is not None and not isinstance(store, RunStore):
        store = RunStore(store)
    return store

def _fixed_profile(profile, backend, variant) -> NoiseProfile:
    """The profile for every mode except "calibrate", which needs the job's counts."""
    if profile == "published":
        return NoiseProfile.from_backend(backend, variant)
    if profile == "ideal":
        return NoiseProfile.from_ideal()
    return profile

def _check_line_qubits(profile, transpiled):
    for tqc in transpiled:
        qubits = line_qubits(tqc)
        if profile.line_qubits is not None and qubits != profile.line_qubits:
            raise ValueError(f"profile was measured on line qubits {profile.line_qubits}, "
                             f"but this run reads out on {qubits}")

def _finish(cases, variant, shots, budget, backend, out, circuits, transpiled, num_extra, profile, known,
            line_qubits, store, tag, job_record) -> list[DetectionResult]:
    """Cutoffs, verdicts and saved records from a job's counts; profile None means fit it from the job."""
    runs = _search_runs(out.counts[num_extra:], out.job_id, shots, circuits, transpiled, backend)
    calibrating = profile is None
    if calibrating:
        profile = profile_from_calibration(backend, out.counts, line_qubits, known)
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

    if store is not None:
        calibration_id = (store.save_calibration(profile, out, backend, known, variant, tag, job_record)
                          if calibrating else None)
        for result in results:
            store.save_detection(result, backend, calibration_id, tag, job_record)
    return results

def detect_winner(board, variant="player", player=None, shots=DEFAULT_SHOTS, budget=DEFAULT_BUDGET,
                  backend=None, profile="auto", store=None) -> DetectionResult:
    """Detect whether anyone has won, using one Grover search plus classical checks.

    Lines measured at least their cutoff are checked on the board; any line
    reported in the result is a real winning line. The default 12-qubit
    "player" variant needs player, the one who just moved; the legacy 22-qubit
    "full" variant checks both players and ignores it. backend defaults to the
    ideal simulator; see detect_winners for profile and store.
    """
    return detect_winners([(board, player)], variant, shots, budget, backend, profile, store)[0]
