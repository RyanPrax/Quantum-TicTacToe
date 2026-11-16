"""The notebooks in notebooks/ run, and are committed with their outputs.

Executing them takes several minutes (noise-model simulations), so those tests
are marked slow. None of them touches real hardware: 03_hardware only submits
with RUN_HARDWARE = True and otherwise reads saved results or rehearses on a
fake backend.
"""

import importlib.util
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted((ROOT / "notebooks").glob("[0-9][0-9]_*.ipynb"))


def _runner():
    spec = importlib.util.spec_from_file_location("run_notebooks", ROOT / "scripts" / "run_notebooks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expected_notebooks():
    assert [p.stem for p in NOTEBOOKS] == ["01_how_it_works", "02_noise", "03_hardware"]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_committed_with_outputs_and_no_errors(path):
    nb = nbformat.read(path, as_version=4)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert code and all(c.execution_count is not None for c in code), "run scripts/run_notebooks.py"
    assert not [o for c in code for o in c.outputs if o.output_type == "error"]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_committed_outputs_come_from_the_current_code(path):
    # scripts/run_notebooks.py stamps the hash of src/qttt it ran with; any change to the package since then
    # may have changed the outputs.
    nb = nbformat.read(path, as_version=4)
    ran_with = nb.metadata.get("qttt", {}).get("source_sha256")
    assert ran_with == _runner().source_hash(), "outputs are stale: run scripts/run_notebooks.py"


def test_hardware_notebook_does_not_submit_by_default():
    nb = nbformat.read(ROOT / "notebooks" / "03_hardware.ipynb", as_version=4)
    assert any("RUN_HARDWARE = False" in c.source for c in nb.cells)


@pytest.mark.slow
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_runs(path):
    _runner().execute(path, save=False)  # raises if any cell fails


def _present_simulated_jobs(store_path):
    """Run 03 on a store of simulated jobs (hardware jobs can't be made in a test); returns its text and Markdown.

    Every analysis cell then runs on real records before any hardware result exists.
    """
    nb = nbformat.read(ROOT / "notebooks" / "03_hardware.ipynb", as_version=4)
    setup = next(c for c in nb.cells if "HARDWARE_ONLY = True" in c.source)
    setup.source = setup.source.replace('STORE = RunStore("../results")', f"STORE = RunStore({str(store_path)!r})")
    setup.source = setup.source.replace("HARDWARE_ONLY = True", "HARDWARE_ONLY = False")
    _runner().run(nb, ROOT / "notebooks")
    return "".join(o.get("text", "") + o.get("data", {}).get("text/markdown", "")
                   for c in nb.cells if c.cell_type == "code" for o in c.outputs)


def test_hardware_notebook_presents_saved_jobs(tmp_path):
    from qttt.backends import IdealBackend
    from qttt.experiments import collect, sample_boards
    from qttt.results import RunStore

    collect(sample_boards(6, seed=1), IdealBackend(seed=1), shots=128, profile="calibrate",
            store=RunStore(tmp_path), boards_per_job=3, tag="test")
    text = _present_simulated_jobs(tmp_path)
    assert text.count("aer ideal") >= 4  # both jobs listed and presented
    assert "No reported line was ever a wrong winner." in text


@pytest.mark.slow
def test_hardware_notebook_compares_with_published_data(tmp_path):
    # A fake IBM device publishes gate errors, so the published-data prediction is drawn too.
    from qttt.backends import get_backend
    from qttt.experiments import collect, sample_boards
    from qttt.results import RunStore

    collect(sample_boards(seed=1, mix={"single_win": 2}), get_backend("fake_berlin", seed=1), shots=64,
            store=RunStore(tmp_path), tag="test")
    from qttt.analysis import jobs

    (job,) = jobs(RunStore(tmp_path), hardware_only=False)
    assert job.published_profile is not None  # so the notebook draws the published-data bars
    text = _present_simulated_jobs(tmp_path)
    assert "ibm fake_berlin" in text and "Sampler options: none" in text
