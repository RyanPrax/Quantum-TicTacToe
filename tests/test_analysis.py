"""Reading saved runs back job by job (qttt.analysis), as 03_hardware does."""

import pytest

from qttt.analysis import Job, describe, examples, jobs, margin, outcome, summarize, verdict, wrong_winners
from qttt.backends import ERROR_SUPPRESSION_OPTIONS, IdealBackend
from qttt.experiments import collect, sample_boards
from qttt.noise import NoiseProfile
from qttt.results import RunStore, profile_summary
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
    return Job([detection], None, {"published_profile": profile_summary(profile)}), profile


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


def test_wilson_interval_always_contains_the_rate():
    # Rounding once put the bound just past p at 0 or n successes (e.g. 6 of 6), which broke found_by_job's whiskers.
    assert all(lo <= k / n <= hi for n in range(1, 300) for k in range(n + 1) for lo, hi in [wilson_interval(k, n)])


def test_rescore_recomputes_verdicts_from_saved_counts(store):
    from qttt.analysis import rescore

    job = jobs(store, hardware_only=False)[0]
    again = rescore(job)
    assert [d["lines"] for d in again.detections] == [d["lines"] for d in job.detections]  # same rule, same verdicts
    assert all(d["lines_at_run"] == d["lines"] and d["cutoffs_at_run"] == d["cutoffs"] for d in again.detections)
    assert again.measured_profile.calibration_id == job.calibration["id"]

    strict = {**job.detections[0], "cutoffs": [10**6] * 8}  # as if an older rule had flagged nothing
    rescored = rescore(Job([strict], job.calibration, job.submitted)).detections[0]
    assert rescored["cutoffs"] == job.detections[0]["cutoffs"] and rescored["cutoffs_at_run"] == [10**6] * 8


def test_a_version_1_job_is_refitted_by_the_current_rule(store):
    import json

    lines = store.runs_file.read_text().splitlines()
    old = []
    for line in lines:
        record = json.loads(line)
        record["schema"] = 1
        if record["kind"] == "detection":  # version 1 kept a full copy, with no-win shares by an older rule
            calibration = next(json.loads(x) for x in lines if json.loads(x)["id"] == record["calibration_id"])
            record["profile"] = {**calibration["profile"], "no_win_shares": [0.5] * 8,
                                 "calibration_id": record["calibration_id"]}
        old.append(json.dumps(record))
    store.runs_file.write_text("\n".join(old) + "\n")
    for job in jobs(RunStore(store.root), hardware_only=False):
        assert job.measured_profile.no_win_shares != (0.5,) * 8
        assert job.measured_profile.calibration_id == job.calibration["id"]


def test_a_detection_that_names_its_calibration_reads_the_profile_from_it(store):
    for job in jobs(store, hardware_only=False):
        assert job.first["profile"] == {"calibration_id": job.calibration["id"]}
        assert job.measured_profile == NoiseProfile.from_record(job.calibration["profile"])


@pytest.mark.parametrize(("true", "found", "expected"), [
    ([], [], "correct"), ([], [3], "wrong"), ([6], [6], "correct"), ([6], [], "missed"),
    ([0, 3], [0], "partly missed"), ([0, 3], [0, 3], "correct"), ([6], [2], "wrong"),
])
def test_verdict(true, found, expected):
    assert verdict({"true_lines": true, "lines": found}) == expected


def test_describe():
    assert describe({"board": "XXXOO    ", "player": "X", "true_lines": [0]}) == "X wins the top row (turn 5)"
    assert describe({"board": "XOX      ", "player": "X", "true_lines": [], "turn": 3}) == \
        "no winner (X just moved, turn 3)"


def _scored(label, true_lines, lines, counts, rejected=()):
    return {"label": label, "true_lines": true_lines, "lines": lines, "rejected": list(rejected),
            "counts": counts, "cutoffs": [100] * 8}


def test_outcome_adds_false_flags():
    assert outcome(_scored("single_win", [6], [6], [0] * 8)) == "correct"
    assert outcome(_scored("single_win", [6], [], [0] * 8, rejected=[2])) == "missed, false flag"


def test_margin_is_the_weakest_winner_or_the_highest_loser():
    assert margin(_scored("double_win", [0, 3], [0], [150, 0, 0, 90, 0, 0, 0, 0])) == 0.9
    assert margin(_scored("no_win", [], [], [80, 120, 0, 0, 0, 0, 0, 0])) == 1.2


def test_examples_pick_the_median_board_of_each_group_correct_ones_first():
    found = [("job", n, _scored("single_win", [6], [6], [0] * 6 + [c, 0])) for n, c in enumerate((300, 200, 150))]
    missed = [("job", 9, _scored("single_win", [6], [], [0] * 6 + [90, 0]))]
    quiet = [("job", 7, _scored("no_win", [], [], [50] * 8))]
    picked = examples(missed + found + quiet, limit=2)
    assert [n for _, n, _ in picked] == [1, 7]  # median found board, then no-win; the missed one is cut
    assert [n for _, n, _ in examples(missed + found + quiet, limit=3)][-1] == 9


@pytest.mark.parametrize(("provider", "name", "label"), [
    ("ibm", "ibm_kingston", "ibm_kingston 2026-11-20"), ("ibm", "fake_berlin", "ibm fake_berlin 2026-11-20"),
    ("aer", "ideal", "aer ideal 2026-11-20"),
])
def test_label_names_the_provider_only_when_the_device_does_not(provider, name, label):
    detection = {"backend": {"provider": provider, "name": name}, "created": "2026-11-20T10:00:00+00:00"}
    assert Job([detection], None, None).label == label


@pytest.mark.parametrize(("options", "text"), [
    ({}, "off"), (None, "off"), (ERROR_SUPPRESSION_OPTIONS, "dynamical decoupling (XY4), gate twirling"),
    ({"twirling": {"enable_gates": False, "enable_measure": True}, "resilience_level": 1},
     "measurement twirling, resilience_level"),
])
def test_suppression_names_the_sampler_options(options, text):
    detection = {"backend": {"options": {"sampler_options": options}}}
    assert Job([detection], None, None).suppression == text
