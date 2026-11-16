"""Permanent, append-only store of every run.

Each run is one JSON object on its own line in <root>/runs.jsonl. Circuits
are saved as <root>/circuits/<id>.qpy: a job's record holds the compiled
template every board in it shares, and each of its detections holds only its
board's few gates, so load_circuit() rebuilds the exact circuit. A job leaves a
"job" record as soon as it is submitted and another with its raw counts as soon
as they are back, so a crash or a lost kernel never loses a job that ran
(finish one with qttt.detector.resume_job). Records are
only ever appended: nothing here edits or deletes a past run. A record made while src/, scripts/ or pyproject.toml
had uncommitted changes (provenance "git_dirty") also names a patch,
<root>/patches/<hash>.patch, holding those changes: check out its "git_commit"
and `git apply` the patch to get back the exact code that made it. Commit the files
to keep them; load them back with RunStore.load() (or pandas.read_json(path,
lines=True)).

Records never contain tokens or account details.

Schema versions (each record's "schema"):
  1  the first records. A profile's no_win_shares were at first each line's
     highest share on a no-win calibration board, later the pooled mean over
     them; the records do not say which.
  2  no_win_shares are each line's pooled share as a loser over every
     calibration board (see qttt.noise.losing_shares), with their standard
     errors and spread. A detection whose cutoffs came from a saved
     calibration record keeps only {"calibration_id": ...} as its profile; the
     calibration record holds the profile itself.
Readers refit a version-1 profile from its calibration counts where they have
them (qttt.analysis.Job.measured_profile).
"""

import hashlib
import json
import os
import platform
import subprocess
import time
import uuid
from datetime import UTC, datetime
from functools import lru_cache
from importlib import metadata
from pathlib import Path

from qiskit import qpy

from qttt.backends import with_board_prep
from qttt.board import O, X, classical_winning_lines

SCHEMA_VERSION = 2
LABELS = ("no_win", "single_win", "double_win")  # by number of true winning lines


def win_label(num_lines: int) -> str:
    """The label for a board with num_lines true winning lines; "multi_win" past LABELS."""
    return LABELS[num_lines] if num_lines < len(LABELS) else "multi_win"


def true_winning_lines(board, player) -> set[int]:
    """The lines really won: by player, or by either player when player is None (legacy variant)."""
    if player is not None:
        return classical_winning_lines(board, player)
    return classical_winning_lines(board, X) | classical_winning_lines(board, O)


class RunStore:
    """Append-only run records under one directory (default: results/)."""

    def __init__(self, root: str | os.PathLike = "results"):
        self.root = Path(root)
        self.runs_file = self.root / "runs.jsonl"
        self.circuits_dir = self.root / "circuits"
        self.patches_dir = self.root / "patches"
        self._records: tuple[tuple[int, int], list[dict]] | None = None  # (file size, mtime) and its parsed records
        self._circuits: dict[str, object] = {}  # circuit file -> loaded circuit

    def save_job_submitted(self, job_id, backend, template, *, cases, variant, shots, budget, known, profile,
                           published=None, tag: str | None = None) -> str:
        """Record a job the moment it is submitted, with everything resume_job needs to finish it.

        template is the backend's transpiled empty-board search circuit; profile is
        the fixed profile the cutoffs will come from, or None when the job calibrates;
        published is the profile from the device's published calibration data at
        submit time, on the same circuit, for comparing with what the job measures.
        """
        record = {
            "kind": "job",
            "status": "submitted",
            "tag": tag,
            "job_id": job_id,
            "variant": variant,
            "shots": shots,
            "budget": budget,
            "cases": [{"board": "".join(board), "player": player} for board, player in cases],
            "known": [{"board": "".join(board), "player": player} for board, player in known],
            "calibrate": profile is None,
            "profile": profile_summary(profile),
            "published_profile": profile_summary(published),
            "backend": _backend_fields(backend),
        }
        return self._append(record, template)

    def save_job_output(self, submitted_id: str | None, output) -> str:
        """Record a job's raw counts as soon as they are back, before anything else can fail."""
        record = {
            "kind": "job",
            "status": "done",
            "submitted_id": submitted_id,
            "job_id": output.job_id,
            "counts": [list(c) for c in output.counts],
        }
        return self._append(record, None)

    def job(self, ref: str) -> tuple[dict, dict | None]:
        """The submitted record of a job, by provider job id or record id, and its done record (None if missing)."""
        jobs = self.load("job")
        submitted = [r for r in jobs if r["status"] == "submitted" and ref in (r["id"], r["job_id"])]
        if not submitted:
            raise KeyError(f"no submitted job {ref!r} in {self.runs_file}")
        submitted = submitted[-1]
        done = [r for r in jobs if r["status"] == "done" and r["submitted_id"] == submitted["id"]]
        return submitted, (done[-1] if done else None)

    def save_detection(self, result, backend, calibration_id: str | None = None, tag: str | None = None,
                       job_record: str | None = None) -> str:
        """Record one detect_winner() result and the backend it ran on; returns the record id.

        calibration_id links the result to the calibration record its cutoffs came
        from; job_record to the job's submitted record; tag names the experiment
        or batch it belongs to. The record also
        holds the board's true label (true_lines, worked out classically), so a
        result can be scored without recomputing anything. A detection in a job
        whose profile is saved in a calibration record (calibration_id) keeps
        only that id as its profile, not another copy of it.
        """
        run = result.run
        true_lines = sorted(true_winning_lines(result.board, result.player))
        record = {
            "kind": "detection",
            "tag": tag,
            "job_record": job_record,
            "calibration_id": calibration_id,
            "variant": result.variant,
            "board": "".join(result.board),
            "player": result.player,
            "turn": 9 - result.board.count(" "),  # marks on the board
            "shots": run.shots,
            "budget": result.budget,
            "cutoffs": list(result.cutoffs),
            "counts": list(run.counts),
            "true_lines": true_lines,
            "label": win_label(len(true_lines)),
            "winner": result.winner,
            "lines": sorted(result.lines),
            "candidates": list(result.candidates),
            "rejected": list(result.rejected),
            "profile": ({"calibration_id": calibration_id} if calibration_id is not None and job_record is not None
                        else profile_summary(result.profile)),
            **_run_fields(run, backend),
        }
        prep = None if run.transpiled is None else (run.transpiled.metadata or {}).get("board_prep")
        if job_record is not None and prep is not None:
            # Every board in a job shares the template saved with the job record: save only this board's gates.
            record["template_record"] = job_record
            record["board_prep"] = prep
            return self._append(record, None)
        return self._append(record, run.transpiled)

    def save_calibration(self, profile, output, backend, known, variant: str = "player", tag: str | None = None,
                         job_record: str | None = None, pooled_from: list[str] | None = None) -> str:
        """Record a calibration: readout counts, known boards and the fitted profile.

        output is the job's output; its first counts are the 8 readout circuits,
        then one per known board (anything after that is ignored). A calibration
        pooled from saved ones (experiments.saved_profile) names their record ids
        in pooled_from and has no job of its own. The profile's template
        fingerprint (backends.template_fingerprint) says which compiled circuit
        it was measured on.
        """
        num_lines = len(profile.readout)
        record = {
            "kind": "calibration",
            "tag": tag,
            "job_record": job_record,
            "variant": variant,
            "shots": sum(output.counts[0]),
            "readout_counts": [list(c) for c in output.counts[:num_lines]],
            "known": [{"board": "".join(board), "player": player, "counts": list(counts)}
                      for (board, player), counts in zip(known, output.counts[num_lines:], strict=False)],
            "profile": profile_summary(profile),
            "job_id": output.job_id,
            "pooled_from": pooled_from,
            "backend": _backend_fields(backend),
        }
        return self._append(record, None)

    def load(self, kind: str | None = None, tag: str | None = None) -> list[dict]:
        """Every saved record, oldest first; optionally only one kind ("detection", "calibration" or "job") or tag.

        The file is parsed again only when it has changed since the last call.
        Each record is a fresh dict, but nested lists and dicts are shared
        between calls: copy them before changing them.
        """
        if not self.runs_file.exists():
            return []
        stat = self.runs_file.stat()
        key = (stat.st_size, stat.st_mtime_ns)
        if self._records is None or self._records[0] != key:
            with self.runs_file.open(encoding="utf-8") as f:
                self._records = (key, [json.loads(line) for line in f if line.strip()])
        return [dict(r) for r in self._records[1]
                if (kind is None or r["kind"] == kind) and (tag is None or r.get("tag") == tag)]

    def load_circuit(self, record: dict):
        """The transpiled circuit saved with a record, or None.

        A detection from a job is rebuilt from the job's template and its own board gates.
        """
        if record.get("template_record"):
            (job,) = [r for r in self.load("job") if r["id"] == record["template_record"]]
            return with_board_prep(self.load_circuit(job), record["board_prep"])
        if not record.get("circuit_file"):
            return None
        if record["circuit_file"] not in self._circuits:  # saved circuits never change
            with (self.root / record["circuit_file"]).open("rb") as f:
                self._circuits[record["circuit_file"]] = qpy.load(f)[0]
        return self._circuits[record["circuit_file"]].copy()

    def _save_patch(self, diff: str) -> str:
        """Save uncommitted code changes once, named by their content, so the records they made can share it."""
        self.patches_dir.mkdir(exist_ok=True)
        path = self.patches_dir / f"{hashlib.sha256(diff.encode()).hexdigest()[:16]}.patch"
        if not path.exists():
            path.write_text(diff, encoding="utf-8")
        return path.relative_to(self.root).as_posix()

    def _append(self, record: dict, circuit) -> str:
        record_id = uuid.uuid4().hex
        record = {
            "id": record_id,
            "schema": SCHEMA_VERSION,
            "created": datetime.now(UTC).isoformat(),
            **record,
            "provenance": provenance(),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        if record["provenance"]["git_dirty"] and _git_cache[1].get("diff"):
            record["provenance"]["patch"] = self._save_patch(_git_cache[1]["diff"])
        if circuit is not None:
            self.circuits_dir.mkdir(exist_ok=True)
            path = self.circuits_dir / f"{record_id}.qpy"
            with path.open("xb") as f:  # "x": never overwrite an existing file
                qpy.dump(circuit, f)
            record["circuit_file"] = path.relative_to(self.root).as_posix()
        line = json.dumps(record, sort_keys=True)
        with self.runs_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        return record_id


GIT_STATE_SECONDS = 10.0  # how long provenance() reuses the git state, so a batch of appends asks git once
_git_cache: tuple[float, dict] | None = None


def provenance() -> dict:
    """Code and library versions, so every record says exactly what produced it.

    The git state is read again after GIT_STATE_SECONDS, so a long notebook
    session records edits made while it runs.
    """
    global _git_cache
    now = time.monotonic()
    if _git_cache is None or now - _git_cache[0] > GIT_STATE_SECONDS:
        _git_cache = (now, _git_state())
    git = {key: value for key, value in _git_cache[1].items() if key != "diff"}
    return {"python": platform.python_version(), "packages": _package_versions(), **git}


@lru_cache(maxsize=1)
def _package_versions() -> dict:
    versions = {}
    for package in ("qttt", "qiskit", "qiskit-aer", "qiskit-ibm-runtime", "numpy"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


CODE_PATHS = (":/src", ":/scripts", ":/pyproject.toml")  # what git_dirty and the patch cover; ":/" is the repo root


def _git_state(here: Path | None = None) -> dict:
    """The commit, whether the code differs from it, and the difference as a patch for `git apply` ("diff").

    Only the code counts (CODE_PATHS), so editing notebooks or docs does not make a run dirty. New files
    that git does not ignore are in the patch too.
    """
    here = here or Path(__file__).resolve().parent

    def git(*args):
        return subprocess.run(["git", *args], cwd=here, capture_output=True, text=True, check=True,
                              timeout=5).stdout

    try:
        commit = git("rev-parse", "HEAD").strip()
        top = git("rev-parse", "--show-toplevel").strip()
        diff = git("diff", "HEAD", "--binary", "--", *CODE_PATHS)
        for name in git("ls-files", "--others", "--exclude-standard", "--full-name", "-z", "--", *CODE_PATHS
                        ).split("\0"):
            if name:  # --no-index exits 1 when the files differ, which a new file always does
                diff += subprocess.run(["git", "diff", "--no-index", "--binary", "--", "/dev/null", name], cwd=top,
                                       capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": diff != "", "diff": diff}


def _backend_fields(backend) -> dict:
    return {
        "name": backend.name,
        "provider": backend.provider,
        "simulated": backend.simulated,
        "calibration_date": backend.calibration_date(),
        "options": backend.options(),
    }


def _run_fields(run, backend) -> dict:
    metrics = run.metrics
    return {
        "job_id": run.job_id,
        "backend": _backend_fields(backend),
        "metrics": None if metrics is None else {
            "two_qubit_gates": metrics.two_qubit_gates,
            "depth": metrics.depth,
            "log10_esp": metrics.log10_esp,
            "log10_idle": metrics.log10_idle,
            "log10_readout": metrics.log10_readout,
            "line_qubits": list(metrics.line_qubits),
            "transpile_seed": metrics.transpile_seed,
            "ancilla": metrics.ancilla,
        },
    }


def profile_summary(profile) -> dict | None:
    """A NoiseProfile as saved in records; NoiseProfile.from_record reads it back."""
    if profile is None:
        return None

    def optional(values):
        return None if values is None else list(values)

    return {
        "source": profile.source,
        "epsilon": profile.epsilon,
        "readout": [list(row) for row in profile.readout],
        "line_qubits": optional(profile.line_qubits),
        "calibration_id": profile.calibration_id,
        "bit_factors": list(profile.bit_factors),
        "no_win_shares": optional(profile.no_win_shares),
        "no_win_shots": profile.no_win_shots,
        "no_win_errors": optional(profile.no_win_errors),
        "no_win_spread": optional(profile.no_win_spread),
        "double_factor": profile.double_factor,
        "calibration_shots": profile.calibration_shots,
        "signal_errors": optional(profile.signal_errors),
        "template": profile.template,
    }
