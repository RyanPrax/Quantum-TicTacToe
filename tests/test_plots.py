"""The notebook figures and tables draw without errors and keep the shared bar style."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from qttt import plots  # noqa: E402
from qttt.board import EMPTY, O, X  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_table_is_markdown():
    md = plots.table(["device", "shots"], [("kingston", 256), ("a|b", 1)], align="lr").data
    assert md.splitlines() == ["| device | shots |", "| :--- | ---: |", "| kingston | 256 |", "| a\\|b | 1 |"]


def test_board_draws_marks_and_labelled_lines():
    fig, ax = plt.subplots()
    plots.draw_board(ax, (X, X, X, X, O, O, X, O, O), [0, 3], "double win")
    texts = [t.get_text() for t in ax.texts]
    assert texts.count("X") == 5 and texts.count("O") == 4
    assert {"0", "3"} <= set(texts)


def test_line_key_has_one_board_per_line():
    assert len(plots.line_key().axes) == 8


def test_rejected_bar_is_hatched_and_marked():
    fig, ax = plt.subplots()
    plots.line_bars(ax, [0.1] * 6 + [0.6, 0.2], winners=[6], rejected=[7], cutoffs=[0.15] * 8, even=1 / 8)
    bars = ax.patches
    assert bars[7].get_hatch() == plots.REJECTED_HATCH and not bars[6].get_hatch()
    assert [t.get_text() for t in ax.texts] == ["x"]


def test_count_grid_scores_each_panel():
    cases = {"Single win": (X, O, X, O, X, O, EMPTY, EMPTY, X)}
    counts = (40, 40, 30, 30, 30, 30, 800, 24)
    fig = plots.count_grid({"ideal": ((150,) * 8, {"Single win": counts})}, cases, {"Single win": [6]}, "t")
    assert "X wins, lines [6], checks: 1" in fig.axes[0].get_title()


def test_found_by_job():
    fig = plots.found_by_job([("job a", 3, 4), ("job b", 0, 0)], "t")
    assert [t.get_text() for t in fig.axes[0].texts] == ["3/4", "0/0"]


def test_board_result_draws_every_line_with_its_share():
    fig, ax = plt.subplots()
    counts = (40, 40, 30, 30, 30, 30, 800, 24)
    plots.board_result(ax, (X, O, X, O, X, O, EMPTY, EMPTY, X), counts, (150,) * 8, winners=[6], rejected=[])
    labels = {t.get_text(): t for t in ax.texts if ":" in t.get_text()}
    assert len(labels) == 8
    assert labels["6: 78%"].get_fontweight() == "bold" and labels["0: 4%"].get_fontweight() == "normal"
    widths = [line.get_linewidth() for line in ax.lines[4:]]  # after the 4 grid lines, thinnest share first
    assert widths == sorted(widths)


def test_board_results_grid():
    board = (X, O, X, O, X, O, EMPTY, EMPTY, X)
    panel = (board, (40, 40, 30, 30, 30, 30, 800, 24), None, [6], [], "t")
    assert sum(ax.axison is False for ax in plots.board_results([panel] * 5, cols=4).axes) == 8
