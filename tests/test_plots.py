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


def _detection(label, true_lines, lines, counts, rejected=(), board="XOXOXO  X"):
    return {"label": label, "board": board, "player": "X", "true_lines": true_lines, "lines": lines,
            "rejected": list(rejected), "candidates": [*lines, *rejected], "counts": counts,
            "cutoffs": [150] * 8, "shots": sum(counts), "turn": 6}


FOUND = _detection("single_win", [6], [6], (40, 40, 30, 30, 30, 30, 800, 24))
MISSED = _detection("single_win", [6], [], (130, 130, 130, 160, 130, 130, 140, 130), rejected=[3])
QUIET = _detection("no_win", [], [], (128,) * 8, board="XOXXOOOXX")


def test_board_details_put_every_board_on_one_scale():
    fig = plots.board_details([("job", 1, FOUND), ("job", 2, MISSED), ("job", 3, QUIET)])
    bars = [ax for ax in fig.axes if ax.axison]
    assert len(bars) == 3 and len({ax.get_ylim() for ax in bars}) == 1
    assert bars[0].get_ylim()[1] >= 800 / 1024
    titles = [sub.get_suptitle() for sub in fig.subfigs]
    assert "job #2: X wins the diagonal from top-left (turn 6)\nfound: nothing (missed, false flag)" in titles


def test_cutoff_margins_count_found_and_flagged_lines():
    fig = plots.cutoff_margins([FOUND, MISSED, QUIET], "t")
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert labels == ["winning lines\nsingle wins\nfound 1 of 2", "losing lines\nboards with a winner\nflagged 1 of 14",
                      "losing lines\nno-win boards\nflagged 0 of 8"]


def test_found_by_line_marks_lines_never_won():
    fig = plots.found_by_line([1, 0, 0, 0, 0, 0, 3, 0], [2, 0, 0, 0, 0, 0, 4, 0], "t")
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert texts.count("none") == 6 and {"1/2", "3/4"} <= set(texts)


def test_predicted_vs_measured_draws_a_dot_per_board_and_model():
    points = {"measured": [(0.3, 0.28, False), (0.5, 0.45, True)], "published": [(0.6, 0.28, False)]}
    fig = plots.predicted_vs_measured(points, "t")
    assert sum(len(c.get_offsets()) for c in fig.axes[0].collections) == 3


def test_readout_diagonals_one_series_per_job():
    from qttt.noise import NoiseProfile

    fig = plots.readout_diagonals({"a": NoiseProfile.from_ideal(), "b": NoiseProfile.from_ideal()}, "t")
    assert len(fig.axes[0].lines) == 2


def test_board_browser_filters_and_steps():
    pytest.importorskip("ipywidgets")
    browser = plots.BoardBrowser([("job b", 1, QUIET), ("job a", 2, MISSED), ("job a", 1, FOUND)])
    assert [r for _, _, r in browser.shown()] == [FOUND, MISSED, QUIET]  # single wins first, best outcome first
    assert browser.status.value == "board 1 of 3"
    browser._step(1)
    browser._step(5)
    assert browser.slider.value == 3
    browser.outcome.value = "missed, false flag"
    assert browser.shown() == [("job a", 2, MISSED)] and browser.slider.value == 1
    browser.outcome.value = "all outcomes"
    browser.job.value = "job b"
    assert browser.status.value == "board 1 of 1"
    browser.kind.value = "single win"
    assert browser.shown() == [] and browser.status.value == "no board matches"
    assert browser._repr_mimebundle_()["text/plain"] == plots.BoardBrowser.STATIC_TEXT
