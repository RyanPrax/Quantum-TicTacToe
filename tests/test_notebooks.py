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
NOTEBOOKS = sorted(p for p in (ROOT / "notebooks").glob("[0-9][0-9]_*.ipynb") if not p.name.startswith("00_"))


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


def test_hardware_notebook_does_not_submit_by_default():
    nb = nbformat.read(ROOT / "notebooks" / "03_hardware.ipynb", as_version=4)
    assert any("RUN_HARDWARE = False" in c.source for c in nb.cells)


@pytest.mark.slow
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_runs(path):
    _runner().execute(path, save=False)  # raises if any cell fails
