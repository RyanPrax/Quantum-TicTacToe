"""Execute the notebooks and save their outputs in place.

    python scripts/run_notebooks.py                  # every numbered notebook except 00
    python scripts/run_notebooks.py 01_how_it_works  # one notebook

Notebooks are committed with their outputs so they read well on GitHub. Run
this after changing a notebook or the code it uses, then commit the result.
"""

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOKS = Path(__file__).resolve().parent.parent / "notebooks"
TIMEOUT = 1800  # seconds per cell


def notebook_paths(names=()):
    if names:
        return [NOTEBOOKS / (n if n.endswith(".ipynb") else f"{n}.ipynb") for n in names]
    return sorted(p for p in NOTEBOOKS.glob("[0-9][0-9]_*.ipynb") if not p.name.startswith("00_"))


def execute(path: Path, save: bool = True) -> nbformat.NotebookNode:
    nb = run(nbformat.read(path, as_version=4), path.parent)
    if save:
        nbformat.write(nb, path)
    return nb


def run(nb: nbformat.NotebookNode, directory: Path) -> nbformat.NotebookNode:
    """Execute a notebook in place, with directory as its working directory; raises if a cell fails."""
    # The kernel talks to this process over TCP on 127.0.0.1 only (Jupyter's default).
    NotebookClient(nb, timeout=TIMEOUT, kernel_name="python3", record_timing=False,
                   resources={"metadata": {"path": str(directory)}}).execute()
    return nb


def main(argv=None):
    for path in notebook_paths(argv if argv is not None else sys.argv[1:]):
        start = time.time()
        execute(path)
        print(f"{path.name}: executed in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
