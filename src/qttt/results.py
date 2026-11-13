"""Permanent, append-only store of every run.

Each run is one JSON object on its own line in <root>/runs.jsonl, and the
transpiled circuit it ran is saved as <root>/circuits/<id>.qpy. Records are
only ever appended: nothing here edits or deletes a past run. Commit the files
to keep them; load them back with RunStore.load() (or pandas.read_json(path,
lines=True)).

Records never contain tokens or account details.
"""

import json
import os
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from importlib import metadata
from pathlib import Path

from qiskit import qpy

from qttt.board import O, X, classical_winning_lines

SCHEMA_VERSION = 1
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

    def save_detection(self, result, backend, calibration_id: str | None = None, tag: str | None = None) -> str:
        """Record one detect_winner() result and the backend it ran on; returns the record id.

        calibration_id links the result to the calibration record its cutoffs came
        from; tag names the experiment or batch it belongs to. The record also
        holds the board's true label (true_lines, worked out classically), so a
        result can be scored without recomputing anything.
        """
        run = result.run
        true_lines = sorted(true_winning_lines(result.board, result.player))
        record = {
            "kind": "detection",
            "tag": tag,
            "calibration_id": calibration_id,
            "variant": result.variant,
            "board": "".join(result.board),
            "player": result.player,
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
            "profile": _profile_summary(result.profile),
            **_run_fields(run, backend),
        }
        return self._append(record, run.transpiled)

    def save_calibration(self, profile, output, backend, known, variant: str = "player", tag: str | None = None) -> str:
        """Record a calibration: readout counts, known boards and the fitted profile.

        output is the job's output; its first counts are the 8 readout circuits,
        then one per known board (anything after that is ignored).
        """
        num_lines = len(profile.readout)
        record = {
            "kind": "calibration",
            "tag": tag,
            "variant": variant,
            "shots": sum(output.counts[0]),
            "readout_counts": [list(c) for c in output.counts[:num_lines]],
            "known": [{"board": "".join(board), "player": player, "counts": list(counts)}
                      for (board, player), counts in zip(known, output.counts[num_lines:])],
            "profile": _profile_summary(profile),
            "job_id": output.job_id,
            "backend": _backend_fields(backend),
        }
        return self._append(record, None)

    def load(self, kind: str | None = None, tag: str | None = None) -> list[dict]:
        """Every saved record, oldest first; optionally only one kind ("detection" or "calibration") or tag."""
        if not self.runs_file.exists():
            return []
        with self.runs_file.open(encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        return [r for r in records if (kind is None or r["kind"] == kind) and (tag is None or r.get("tag") == tag)]

    def load_circuit(self, record: dict):
        """The transpiled circuit saved with a record, or None."""
        if not record.get("circuit_file"):
            return None
        with (self.root / record["circuit_file"]).open("rb") as f:
            return qpy.load(f)[0]

    def _append(self, record: dict, circuit) -> str:
        record_id = uuid.uuid4().hex
        record = {
            "id": record_id,
            "schema": SCHEMA_VERSION,
            "created": datetime.now(timezone.utc).isoformat(),
            **record,
            "provenance": provenance(),
        }
        self.root.mkdir(parents=True, exist_ok=True)
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


@lru_cache(maxsize=1)
def provenance() -> dict:
    """Code and library versions, so every record says exactly what produced it."""
    versions = {}
    for package in ("qttt", "qiskit", "qiskit-aer", "qiskit-ibm-runtime"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "packages": versions, **_git_state()}


def _git_state(here: Path = Path(__file__).resolve().parent) -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True,
                                text=True, check=True, timeout=5).stdout.strip()
        # ":/src" is src/ at the repository root; a plain "src" would be read relative to here.
        dirty = subprocess.run(["git", "status", "--porcelain", "--", ":/src"], cwd=here, capture_output=True,
                               text=True, check=True, timeout=5).stdout.strip() != ""
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": dirty}


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
            "line_qubits": list(metrics.line_qubits),
            "transpile_seed": metrics.transpile_seed,
        },
    }


def _profile_summary(profile) -> dict | None:
    if profile is None:
        return None
    return {
        "source": profile.source,
        "epsilon": profile.epsilon,
        "readout": [list(row) for row in profile.readout],
        "line_qubits": None if profile.line_qubits is None else list(profile.line_qubits),
    }
