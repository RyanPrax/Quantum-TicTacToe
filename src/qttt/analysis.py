"""Read saved runs back from a RunStore job by job, for 03_hardware.

Everything comes from the records themselves (provider, device, calibration
date, published data), so this works for any device and any provider; a
provider that publishes no gate errors simply has no published-data profile.
"""

from collections import defaultdict
from dataclasses import dataclass

from qttt.noise import NoiseProfile
from qttt.results import LABELS, RunStore
from qttt.stats import wilson_interval


@dataclass(frozen=True)
class Job:
    """One job's records: its detections, the calibration fitted in it, and its submitted record."""
    detections: list[dict]
    calibration: dict | None
    submitted: dict | None  # None for records saved before jobs were recorded

    @property
    def first(self) -> dict:
        return self.detections[0]

    @property
    def backend(self) -> dict:
        return self.first["backend"]

    @property
    def label(self) -> str:
        """Provider, device and day, e.g. "ibm ibm_kingston 2026-11-20"."""
        return f"{self.backend['provider']} {self.backend['name']} {self.first['created'][:10]}"

    @property
    def measured_profile(self) -> NoiseProfile:
        """The profile the job's cutoffs came from (fitted in the job when it calibrated).

        Read from a detection, whose copy carries the calibration record's id.
        """
        return NoiseProfile.from_record(self.first["profile"])

    @property
    def published_profile(self) -> NoiseProfile | None:
        """The profile from the device's published calibration data at submit time, if it publishes any."""
        if self.submitted is None or self.submitted.get("published_profile") is None:
            return None
        if self.first["metrics"]["log10_esp"] is None:  # no published gate errors: nothing to compare with
            return None
        return NoiseProfile.from_record(self.submitted["published_profile"])


def jobs(store: RunStore, hardware_only: bool = True) -> list[Job]:
    """Every job in the store, oldest first; by default only jobs that ran on real hardware."""
    detections = [r for r in store.load("detection") if not (hardware_only and r["backend"]["simulated"])]
    groups = defaultdict(list)
    for r in detections:
        groups[r.get("job_record") or r["job_id"] or r["id"]].append(r)
    calibrations = {r["id"]: r for r in store.load("calibration")}
    submitted = {r["id"]: r for r in store.load("job") if r["status"] == "submitted"}
    found = [Job(group, calibrations.get(group[0]["calibration_id"]), submitted.get(group[0].get("job_record")))
             for group in groups.values()]
    return sorted(found, key=lambda job: job.first["created"])


@dataclass(frozen=True)
class LabelSummary:
    """How one label's boards did in a set of detections."""
    label: str
    boards: int
    wins_found: int  # boards where at least one winning line was confirmed
    all_lines_found: int  # boards where every winning line was confirmed
    false_flags: int  # rejected classical checks
    checks: int  # classical checks used

    @property
    def found_interval(self) -> tuple[float, float]:
        """95% Wilson interval for the share of boards whose win was found."""
        return wilson_interval(self.wins_found, self.boards)


def summarize(detections: list[dict]) -> list[LabelSummary]:
    """Per label (no win, single win, double win), the counts behind the summary table."""
    rows = []
    for label in LABELS:
        group = [r for r in detections if r["label"] == label]
        if group:
            rows.append(LabelSummary(
                label=label,
                boards=len(group),
                wins_found=sum(bool(r["lines"]) for r in group),
                all_lines_found=sum(sorted(r["lines"]) == r["true_lines"] for r in group),
                false_flags=sum(len(r["rejected"]) for r in group),
                checks=sum(len(r["candidates"]) for r in group),
            ))
    return rows


def wrong_winners(detections: list[dict]) -> list[dict]:
    """Detections that reported a line that is not a real win (the classical check makes this impossible)."""
    return [r for r in detections if not set(r["lines"]) <= set(r["true_lines"])]
