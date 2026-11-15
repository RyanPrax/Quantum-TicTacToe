"""Reading saved runs back job by job (qttt.analysis), as 03_hardware does."""

import pytest

from qttt.analysis import Job, jobs, summarize, wrong_winners
from qttt.backends import IdealBackend
from qttt.experiments import collect, sample_boards
from qttt.noise import NoiseProfile
from qttt.results import RunStore, _profile_summary
from qttt.stats import wilson_interval

SEED = 4903


@pytest.fixture
def store(tmp_path):
    store = RunStore(tmp_path)
    collect(sample_boards(6, seed=SEED), IdealBackend(seed=SEED), shots=256, profile="calibrate", store=store,
            boards_per_job=3, tag="two-jobs")
    return store


def test_simulated_runs_are_not_hardware_jobs(store):
    assert jobs(store) == []


def test_records_group_into_their_jobs(store):
    found = jobs(store, hardware_only=False)
    assert [len(job.detections) for job in found] == [3, 3]
    for job in found:
        assert job.submitted["id"] == job.first["job_record"]
        assert job.calibration["id"] == job.first["calibration_id"]
        assert job.measured_profile.calibration_id == job.calibration["id"]
        assert job.label.startswith("aer ideal ")
        assert job.published_profile is None  # the ideal simulator publishes no gate errors


def test_summary_counts_and_no_wrong_winner(store):
    detections = [d for job in jobs(store, hardware_only=False) for d in job.detections]
    rows = {row.label: row for row in summarize(detections)}
    assert sum(row.boards for row in rows.values()) == 6
    assert rows["single_win"].wins_found == rows["single_win"].boards  # a perfect simulator finds every win
    assert wrong_winners(detections) == []
    assert wrong_winners([{**detections[0], "lines": [9], "true_lines": []}])


def _job_with(log10_esp):
    profile = NoiseProfile(0.3, NoiseProfile.from_ideal().readout, "backend:test", line_qubits=(1, 2, 3))
    detection = {"metrics": {"log10_esp": log10_esp}, "backend": {"provider": "x", "name": "y"}, "created": "2026"}
    return Job([detection], None, {"published_profile": _profile_summary(profile)}), profile


def test_published_profile_only_when_the_provider_publishes_gate_errors():
    job, profile = _job_with(-0.3)
    assert job.published_profile == profile
    job, _ = _job_with(None)
    assert job.published_profile is None


@pytest.mark.parametrize("k, n, low, high", [(30, 30, 0.886, 1.0), (0, 30, 0.0, 0.114), (5, 10, 0.237, 0.763)])
def test_wilson_interval(k, n, low, high):
    lo, hi = wilson_interval(k, n)
    assert lo == pytest.approx(low, abs=1e-3) and hi == pytest.approx(high, abs=1e-3)
    assert wilson_interval(0, 0) == (0.0, 1.0)
