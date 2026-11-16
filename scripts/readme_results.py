"""Regenerate the README's headline figure and results table from the code and the saved results.

    python scripts/readme_results.py

Writes figures/circuit_versions.png and rewrites the block between the
"results:start" and "results:end" markers in README.md. The numbers are the
ones the notebooks compute: the circuit comparison from 01_how_it_works
(compiled again, with the same device and seed), the noise-model results that
02_noise saves to results/noise_summary.json when it runs, and the hardware
jobs from 03_hardware. Takes seconds. Run it after the notebooks change or a
hardware job is saved, then commit README.md and the figure.
"""

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from qttt import plots  # noqa: E402
from qttt.analysis import jobs, rescore, wrong_winners  # noqa: E402
from qttt.backends import get_backend  # noqa: E402
from qttt.board import EMPTY, X  # noqa: E402
from qttt.grover import build_phase_kickback_circuit, build_search_circuit  # noqa: E402
from qttt.results import RunStore  # noqa: E402
from qttt.stats import wilson_interval  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
FIGURE = ROOT / "figures" / "circuit_versions.png"
NOISE_SUMMARY = ROOT / "results" / "noise_summary.json"
SEED = 4903
NOTEBOOK = "notebooks/{}.ipynb"


def circuit_row():
    kingston = get_backend("fake_kingston", seed=SEED, transpile_seeds=5)
    empty = [EMPTY] * 9
    versions = {
        "original: 2 qubits per cell": build_search_circuit("full", empty),
        "1 qubit per cell + phase qubit": build_phase_kickback_circuit(empty, X),
        "1 qubit per cell, direct sign flip": build_search_circuit("player", empty, X),
    }
    rows = []
    for name, qc in versions.items():
        m = kingston.metrics(kingston.transpile(qc))
        rows.append((name, qc.num_qubits, m.two_qubit_gates, 10 ** m.log10_esp))
    (_, q0, g0, e0), (_, q1, g1, e1) = rows[0], rows[-1]
    fig = plots.circuit_versions(rows, f"{q0} → {q1} qubits: {g0 / g1:.1f}× fewer two-qubit gates, "
                                       f"ESP {e0:.1%} → {e1:.0%} (compiled for fake_kingston)")
    FIGURE.parent.mkdir(exist_ok=True)
    fig.savefig(FIGURE, dpi=120, bbox_inches="tight")
    return (f"{q0} → {q1} qubits, {g0} → {g1} two-qubit gates ({g0 / g1:.1f}× fewer), "
            f"chance no gate fails (ESP) {e0:.1%} → {e1:.1%} on a simulated ibm_kingston")


def noise_row():
    """From the numbers 02_noise saves when it runs, so the README shows exactly what the notebook found."""
    if not NOISE_SUMMARY.exists():
        sys.exit(f"{NOISE_SUMMARY.relative_to(ROOT)} is missing: run notebooks/02_noise.ipynb first")
    devices = json.loads(NOISE_SUMMARY.read_text())["devices"]
    parts, gaps = [], []
    for label, d in devices.items():
        gaps.append(d["epsilon_measured"] - d["epsilon_published"])
        parts.append(f"{label} {d['shots_needed_measured'] or 'more than 16384'} (published data: "
                     f"{d['shots_needed_published'] or 'more than 16384'})")
    return (f"Signal survives on all 3 free-plan devices, but measured noise is {min(gaps):.2f}–{max(gaps):.2f} "
            f"above IBM's published figures; shots for ≤1% missed wins: {', '.join(parts)}")


def hardware_row():
    saved = [rescore(job) for job in jobs(RunStore(ROOT / "results"))]
    if not saved:
        return "No hardware job saved yet"
    detections = [d for job in saved for d in job.detections]
    with_winner = [d for d in detections if d["true_lines"]]
    found = sum(bool(d["lines"]) for d in with_winner)
    low, high = wilson_interval(found, len(with_winner))
    assert not wrong_winners(detections)
    devices = ", ".join(sorted({job.backend["name"] for job in saved}))
    suppression = {job.suppression for job in saved}
    suppression = (f"error suppression {suppression.pop()}" if len(suppression) == 1
                   else "error suppression differs between jobs")
    return (f"{len(detections)} boards on {devices} ({suppression}): wins found on {found} of {len(with_winner)} "
            f"(95% interval {low:.0%}–{high:.0%}); no wrong winner reported")


def main():
    rows = [("Circuit", circuit_row(), "01_how_it_works"), ("Simulated noise", noise_row(), "02_noise"),
            ("Real hardware", hardware_row(), "03_hardware")]
    block = "\n".join(["| | Result | Notebook |", "| :--- | :--- | :--- |"]
                      + [f"| **{stage}** | {result} | [{nb}]({NOTEBOOK.format(nb)}) |" for stage, result, nb in rows])
    text = README.read_text()
    new, n = re.subn(r"(<!-- results:start -->\n).*?(<!-- results:end -->)", lambda m: m[1] + block + "\n" + m[2],
                     text, flags=re.S)
    if n != 1:
        sys.exit("README.md needs exactly one results:start / results:end block")
    README.write_text(new)
    print(block)


if __name__ == "__main__":
    main()
