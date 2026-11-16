"""Figures and tables for the notebooks and the README (needs the notebooks extra: matplotlib, IPython).

Bars for the 8 line indices share one style everywhere: green for a real
winning line, gray for a losing line, and orange with hatching and an "x" for
a losing line that reached its cutoff and was rejected by the classical check
(the hatching and the mark keep it readable without colour).
"""

import math
from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from qttt.board import EMPTY, SHORT_LINE_NAMES, WIN_LINES
from qttt.detector import NUM_LINES, flag_and_check

WIN = "tab:green"
LOSE = "tab:gray"
REJECTED = "tab:orange"
REJECTED_HATCH = "///"
MARK_COLORS = {"X": "tab:blue", "O": "tab:red"}


def table(header: Sequence[str], rows: Sequence[Sequence], align: str | None = None):
    """A Markdown table (renders as a real table on GitHub). align: one "l" or "r" per column, default all "l"."""
    from IPython.display import Markdown

    align = align or "l" * len(header)
    rule = ["---:" if a == "r" else ":---" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(rule) + " |"]
    lines += ["| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |" for row in rows]
    return Markdown("\n".join(lines))


def markdown(text: str):
    from IPython.display import Markdown

    return Markdown(text)


# --- Boards -------------------------------------------------------------------------------------------------


def _centre(cell: int) -> tuple[float, float]:
    """Plot coordinates of a cell's centre: cell 0 top left, 8 bottom right, each cell 1 wide."""
    row, col = divmod(cell, 3)
    return col + 0.5, 2.5 - row


def draw_board(ax, board: Sequence[str], lines: Sequence[int] = (), title: str | None = None,
               label_lines: bool = True, fontsize: float = 18) -> None:
    """A 3x3 grid with its marks; each line in lines is drawn through its cells, labelled with its index."""
    pad = 0.3 if label_lines and lines else 0.05
    ax.set_xlim(-pad, 3 + pad)
    ax.set_ylim(-pad, 3 + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    for k in (1, 2):
        ax.plot([k, k], [0, 3], color="black", linewidth=1.2)
        ax.plot([0, 3], [k, k], color="black", linewidth=1.2)

    for cell, mark in enumerate(board):
        if mark != EMPTY:
            ax.text(*_centre(cell), mark, ha="center", va="center", fontsize=fontsize, fontweight="bold",
                    color=MARK_COLORS.get(mark, "black"))
    for line in lines:
        (x0, y0), (x1, y1) = _centre(WIN_LINES[line][0]), _centre(WIN_LINES[line][2])
        ax.plot([x0, x1], [y0, y1], color=WIN, linewidth=7, alpha=0.45, solid_capstyle="round")
        if label_lines:  # the index sits where the line leaves its last cell
            dx, dy = np.sign(x1 - x0), np.sign(y1 - y0)
            ax.text(x1 + 0.36 * dx, y1 + 0.36 * dy, str(line), ha="center", va="center", fontsize=fontsize * 0.5,
                    fontweight="bold", color="white",
                    bbox={"boxstyle": "circle,pad=0.25", "facecolor": WIN, "edgecolor": "none"})
    if title:
        ax.set_title(title, fontsize=9)


def show_boards(boards: Mapping[str, tuple[Sequence[str], Sequence[int]]], size: float = 2.0):
    """A row of boards. boards: {title: (board, winning lines)}."""
    fig, axes = plt.subplots(1, len(boards), figsize=(size * len(boards), size * 1.1), squeeze=False)
    for ax, (title, (board, lines)) in zip(axes[0], boards.items(), strict=True):
        draw_board(ax, board, sorted(lines), title)
    fig.tight_layout()
    return fig


# Where each line's share is written: rows to the right, columns below, diagonals past their bottom corner.
_SHARE_LABELS = ((3.4, 2.5), (3.4, 1.5), (3.4, 0.5), (0.5, -0.3), (1.5, -0.3), (2.5, -0.3), (3.4, -0.3), (-0.4, -0.3))


def board_result(ax, board: Sequence[str], counts: Sequence[int], cutoffs: Sequence[int] | None = None,
                 winners=(), rejected=(), title: str | None = None, shares: bool = True) -> None:
    """A board with the circuit's output drawn on it: every line through its cells, as thick as its share of shots.

    Green: a real winning line. Orange, dashed: flagged (reached its cutoff) and rejected by the check. Gray: a
    losing line. Each line's index (and its share, when shares is True) is written where it leaves the board, in
    bold when it reached its cutoff.
    """
    winners, rejected = set(winners), set(rejected)
    shots = sum(counts)
    draw_board(ax, board, fontsize=14)
    for t in ax.texts:  # the marks sit on top of the lines, outlined so they stay readable
        t.set_zorder(5)
        t.set_path_effects([patheffects.withStroke(linewidth=3, foreground="white")])
    ax.set_xlim(-0.8, 3.8)
    ax.set_ylim(-0.6, 3.25)
    for i, cells in sorted(enumerate(WIN_LINES), key=lambda item: counts[item[0]]):  # biggest share on top
        share = counts[i] / shots
        color = WIN if i in winners else REJECTED if i in rejected else LOSE
        (x0, y0), (x1, y1) = _centre(cells[0]), _centre(cells[2])
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=1 + 30 * share, alpha=0.6 if i in winners else 0.45,
                linestyle="--" if i in rejected else "-", solid_capstyle="round", zorder=2)
        flagged = cutoffs is not None and counts[i] >= cutoffs[i]
        ax.text(*_SHARE_LABELS[i], f"{i}: {share:.0%}" if shares else str(i), ha="center", va="center",
                fontsize=8, fontweight="bold" if flagged else "normal", color=color if color != LOSE else "dimgray")
    if title:
        ax.set_title(title, fontsize=8)


def board_results(panels: Sequence[tuple[Sequence[str], Sequence[int], Sequence[int] | None, Sequence[int],
                                         Sequence[int], str]], title: str | None = None, cols: int = 4):
    """A grid of board_result panels: (board, counts, cutoffs, winning lines, rejected lines, panel title)."""
    rows = -(-len(panels) // cols)
    cols = min(cols, len(panels))
    fig, axes = plt.subplots(rows, cols, figsize=(2.9 * cols, 2.9 * rows + 0.6), squeeze=False, layout="constrained")
    for ax, (board, counts, cutoffs, winners, rejected, panel) in zip(axes.flat, panels, strict=False):
        board_result(ax, board, counts, cutoffs, winners, rejected, panel)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.legend(handles=_board_handles(), loc="outside lower center", ncol=3, fontsize=8, frameon=False,
               title="thickness: share of shots; bold: reached its cutoff", title_fontsize=8)
    return fig


def _board_handles() -> list:
    return [Line2D([], [], color=WIN, linewidth=5, alpha=0.6, label="winning line"),
            Line2D([], [], color=LOSE, linewidth=5, alpha=0.45, label="losing line"),
            Line2D([], [], color=REJECTED, linewidth=3, linestyle="--", label="flagged, rejected by the check")]


def line_key(bits: bool = True):
    """The 8 line indices, each drawn on an empty board (with its 3-qubit code when bits is True)."""
    fig, axes = plt.subplots(1, NUM_LINES, figsize=(1.45 * NUM_LINES, 1.9))
    for i, ax in enumerate(axes):
        title = f"{i}: {SHORT_LINE_NAMES[i]}" + (f"\nqubits {i:03b}" if bits else "")
        draw_board(ax, [EMPTY] * 9, [i], label_lines=False)
        ax.set_title(title, fontsize=7)
    fig.tight_layout()
    return fig


# --- Bars over the 8 line indices ---------------------------------------------------------------------------


def line_bars(ax, values: Sequence[float], winners=(), rejected=(), cutoffs: Sequence[float] | None = None,
              even: float | None = None) -> None:
    """One bar per line index in the shared style; cutoffs draws a black tick on each bar, even a dotted line."""
    winners, rejected = set(winners), set(rejected)
    for i, v in enumerate(values):
        style = {"color": WIN} if i in winners else {
            "color": REJECTED, "hatch": REJECTED_HATCH, "edgecolor": "white", "linewidth": 0} if i in rejected else {
            "color": LOSE}
        ax.bar(i, v, **style)
        if i in rejected:
            ax.text(i, v, "x", ha="center", va="bottom", fontsize=8, fontweight="bold", color=REJECTED)
    if cutoffs is not None:
        ax.hlines(cutoffs, [i - 0.4 for i in range(NUM_LINES)], [i + 0.4 for i in range(NUM_LINES)],
                  color="black", linewidth=1.5)
    if even is not None:
        ax.axhline(even, color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(range(NUM_LINES))


def line_legend(fig, cutoffs: bool = True, rejected: bool = True, even: bool = True) -> float:
    """The shared legend under the figure; returns the share of the figure's height it takes (for tight_layout)."""
    handles = _bar_handles(cutoffs, rejected, even)
    width, height = fig.get_size_inches()
    ncol = len(handles) if width >= 11 else 3
    fig.legend(handles=handles, loc="lower center", ncol=ncol, fontsize=8, frameon=False)
    return 0.3 * -(-len(handles) // ncol) / height


def _bar_handles(cutoffs: bool = True, rejected: bool = True, even: bool = True) -> list:
    handles = [Patch(color=WIN, label="winning line"), Patch(color=LOSE, label="losing line")]
    if rejected:
        handles.append(Patch(facecolor=REJECTED, hatch=REJECTED_HATCH, edgecolor="white",
                             label="x: flagged, rejected by the check"))
    if cutoffs:
        handles.append(Line2D([], [], color="black", linewidth=1.5, label="cutoff"))
    if even:
        handles.append(Line2D([], [], color="gray", linestyle=":", label="1/8 (even spread)"))
    return handles


def probability_bars(panels: Mapping[str, tuple[Sequence[float], Sequence[int]]], ylabel: str = "probability",
                     title: str | None = None, mean: tuple[int, float] | None = None):
    """A row of line-index bar charts. panels: {title: (values, winning lines)}; mean: (panel, value) to mark."""
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 2.8), sharey=True, squeeze=False)
    for k, (ax, (name, (values, winners))) in enumerate(zip(axes[0], panels.items(), strict=True)):
        line_bars(ax, values, winners)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("line index")
        if mean is not None and mean[0] == k:
            ax.axhline(mean[1], color="tab:blue", linestyle="--", linewidth=1, label=f"mean {mean[1]:.3f}")
            ax.legend(fontsize=8)
    axes[0][0].set_ylabel(ylabel)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def count_grid(columns: Mapping[str, tuple[Sequence[int], Mapping[str, Sequence[int]]]],
               cases: Mapping[str, Sequence[str]], winners: Mapping[str, Sequence[int]], title: str):
    """Rows: cases; columns: {column title: (cutoffs, {case name: counts})}. Each panel scored as a real run."""
    fig, axes = plt.subplots(len(cases), len(columns), figsize=(3.2 * len(columns), 2.8 * len(cases)),
                             squeeze=False, sharey=True)
    for c, (col, (cutoffs, by_case)) in enumerate(columns.items()):
        for r, (name, board) in enumerate(cases.items()):
            ax, counts = axes[r][c], by_case[name]
            verdict = flag_and_check(board, counts, cutoffs)
            shots = sum(counts)
            line_bars(ax, [n / shots for n in counts], winners[name], verdict.rejected,
                      [k / shots for k in cutoffs], even=1 / 8)
            ax.set_ylim(0, 1)
            text = f"{verdict.winner} wins, lines {sorted(verdict.lines)}" if verdict.winner else "no winner found"
            ax.set_title(f"{col if r == 0 else ''}\n{text}, checks: {len(verdict.candidates)}", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{name}\nfraction of shots")
            if r == len(cases) - 1:
                ax.set_xlabel("line index")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, line_legend(fig), 1, 1))
    return fig


# --- 01_how_it_works ----------------------------------------------------------------------------------------


def circuit_versions(rows: Sequence[tuple[str, int, int, float]], title: str):
    """Two-qubit gates and ESP per circuit version. rows: (name, qubits, two-qubit gates, ESP)."""
    names = [f"{name}\n{qubits} qubits" for name, qubits, _, _ in rows]
    y = np.arange(len(rows))[::-1]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 0.75 * len(rows) + 1.3), sharey=True)
    colors = [LOSE] * (len(rows) - 1) + [WIN]
    gates = [g for _, _, g, _ in rows]
    esp = [100 * e for _, _, _, e in rows]
    left.barh(y, gates, color=colors)
    right.barh(y, esp, color=colors)
    for v, yy in zip(gates, y, strict=True):
        left.text(v, yy, f" {v}", va="center", fontsize=8)
    for v, yy in zip(esp, y, strict=True):
        right.text(v, yy, f" {v:.1f}%", va="center", fontsize=8)
    left.set_yticks(y, names, fontsize=8)
    left.set_xlabel("two-qubit gates after compiling (fewer is better)")
    right.set_xlabel("ESP: chance no gate fails, % (higher is better)")
    left.set_xlim(0, max(gates) * 1.18)
    right.set_xlim(0, 100)
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def grover_iterations(max_iterations: int = 3):
    """Total probability on the winning lines after k Grover iterations, M = 1 and M = 2 of 8 marked."""
    ks = np.arange(max_iterations + 1)
    fig, ax = plt.subplots(figsize=(7.5, 3))
    for m, style in ((1, "o-"), (2, "s--")):
        ax.plot(ks, np.sin((2 * ks + 1) * np.arcsin(np.sqrt(m / 8))) ** 2, style, label=f"M = {m} winning line(s)")
    ax.axhline(2 / 8, color="gray", linestyle=":", linewidth=1, label="M = 2 by chance (2 of 8)")
    ax.axvline(1, color=WIN, alpha=0.25, linewidth=8)
    ax.set_xticks(ks)
    ax.set_xlabel("Grover iterations")
    ax.set_ylabel("probability on winning lines")
    ax.set_title("1 iteration is the only count that separates single, double and no win", fontsize=9)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    return fig


def grover_rotation():
    """The state as a vector in the (|bad>, |good>) plane after k = 0, 1, 2 iterations, M = 1 and M = 2."""
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="lightgray"))
    for m, color in ((1, "tab:blue"), (2, "tab:orange")):
        theta = math.asin(math.sqrt(m / 8))
        for k, alpha in ((0, 0.35), (1, 1.0), (2, 0.35)):
            angle = (2 * k + 1) * theta
            ax.annotate("", xy=(math.cos(angle), math.sin(angle)), xytext=(0, 0),
                        arrowprops={"arrowstyle": "->", "color": color, "alpha": alpha, "linewidth": 2})
            side = math.cos(angle)  # labels left of the vertical end at the tip, labels right of it start there
            ha, va = ("center", "bottom") if abs(side) < 0.1 else ("left" if side > 0 else "right", "center")
            ax.text(1.06 * math.cos(angle), 1.06 * math.sin(angle), f"M={m}, k={k}", color=color,
                    alpha=max(alpha, 0.6), fontsize=8, ha=ha, va=va)
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-0.2, 1.25)
    ax.set_aspect("equal")
    ax.set_xlabel("|bad⟩ (losing lines)")
    ax.set_ylabel("|good⟩ (winning lines)")
    ax.set_title("Each iteration rotates the state 2θ toward |good⟩", fontsize=9)
    fig.tight_layout()
    return fig


def count_distributions(shots: int, cutoff: int, pmf):
    """Distribution of one line's count for a losing line, a single winner and each of two winners."""
    ks = np.arange(0, int(0.93 * shots))
    fig, ax = plt.subplots(figsize=(7, 2.8))
    for p, label in ((1 / 8, "losing line (1/8)"), (25 / 32, "single winner (25/32)"),
                     (1 / 2, "each of two winners (1/2)")):
        ax.plot(ks, [pmf(int(k), shots, p) for k in ks], label=label)
    ax.axvline(cutoff, color="black", linestyle="--", linewidth=1, label=f"cutoff {cutoff}")
    ax.set_xlim(int(0.06 * shots), ks[-1])
    ax.set_xlabel(f"count out of {shots} shots")
    ax.set_ylabel("probability")
    ax.set_title("On a perfect simulator the cutoff sits far from every winner's count", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# --- 02_noise -----------------------------------------------------------------------------------------------


def miss_rate_curves(curves: Mapping[str, tuple[Sequence[float], Sequence[float]]], shot_counts: Sequence[int],
                     title: str):
    """Worst chance of missing a win against shots. curves: {device: (measured, published)}."""
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for (label, (measured, published)), color in zip(curves.items(), ("tab:blue", "tab:orange", "tab:purple"),
                                                     strict=False):
        ax.plot(shot_counts, [max(v, 1e-12) for v in measured], "o-", color=color, label=f"{label} (measured)")
        ax.plot(shot_counts, [max(v, 1e-12) for v in published], "s--", color=color, alpha=0.5,
                label=f"{label} (published)")
    ax.axhline(0.01, color="gray", linestyle=":", linewidth=1, label="1% misses")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(1e-12, 1.5)
    ax.set_xlabel("shots per board")
    ax.set_ylabel("worst chance of missing a win")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


# --- 03_hardware --------------------------------------------------------------------------------------------


def found_by_job(rows: Sequence[tuple[str, int, int]], title: str):
    """Share of boards with a winner whose win was found, per job, with 95% Wilson intervals.

    rows: (job label, wins found, boards with a winner).
    """
    from qttt.stats import wilson_interval

    fig, ax = plt.subplots(figsize=(6.5, 0.6 * len(rows) + 1.4))
    y = np.arange(len(rows))[::-1]
    for (_, found, boards), yy in zip(rows, y, strict=True):
        low, high = wilson_interval(found, boards)
        share = found / boards if boards else 0
        ax.barh(yy, share, color=WIN, alpha=0.8)
        ax.errorbar(share, yy, xerr=[[share - low], [high - share]], color="black", capsize=4)
        ax.text(1.01, yy, f"{found}/{boards}", va="center", fontsize=8)
    ax.set_yticks(y, [label for label, _, _ in rows], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of wins found (bar), 95% interval (whiskers)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


def readout_diagonals(profiles: Mapping[str, object], title: str):
    """How often each prepared line index is read back correctly (readout matrix A's diagonal), per job.

    profiles: {job label: NoiseProfile}.
    """
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    offsets = np.linspace(-0.25, 0.25, len(profiles)) if len(profiles) > 1 else [0.0]
    low = 1.0
    for offset, (label, profile) in zip(offsets, profiles.items(), strict=True):
        diagonal = [profile.readout[i][i] for i in range(NUM_LINES)]
        low = min(low, *diagonal)
        ax.plot(np.arange(NUM_LINES) + offset, diagonal, "o", markersize=5, label=label)
    ax.set_xticks(range(NUM_LINES))
    ax.set_ylim(min(0.9, low - 0.01), 1.0)
    ax.set_xlabel("prepared line index")
    ax.set_ylabel("read back correctly")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    return fig


def _shares(detection: dict) -> tuple[list[float], list[float]]:
    """A saved board's share of shots on each line index, and each line's cutoff as a share."""
    shots = detection["shots"]
    return [c / shots for c in detection["counts"]], [k / shots for k in detection["cutoffs"]]


def _caption(job: str, number: int, detection: dict) -> str:
    from qttt.analysis import describe, outcome

    found = " and ".join(SHORT_LINE_NAMES[i] for i in detection["lines"]) or "nothing"
    return f"{job} #{number}: {describe(detection)}\nfound: {found} ({outcome(detection)})"


def board_detail(fig, detection: dict, title: str | None = None, top: float | None = None) -> None:
    """One saved board on a figure (or subfigure): the board with its output drawn on it, and its bars.

    The bars use the shared style, with each line's cutoff and 1/8 marked; top sets the bars' y-axis limit.
    """
    board_ax, bars_ax = fig.subplots(1, 2, width_ratios=(1, 1.25))
    shares, cutoffs = _shares(detection)
    board_result(board_ax, tuple(detection["board"]), detection["counts"], detection["cutoffs"],
                 detection["true_lines"], detection["rejected"], shares=False)
    line_bars(bars_ax, shares, detection["true_lines"], detection["rejected"], cutoffs, even=1 / 8)
    bars_ax.set_ylim(0, top or _bars_top([detection]))
    bars_ax.set_xlabel("line index", fontsize=8)
    bars_ax.set_ylabel("share of shots", fontsize=8)
    bars_ax.tick_params(labelsize=7)
    bars_ax.spines[["top", "right"]].set_visible(False)
    if title:
        fig.suptitle(title, fontsize=8)


def _bars_top(detections: Sequence[dict]) -> float:
    """A y-axis limit that fits every bar and cutoff of these boards, rounded up to 0.05 (at least 0.3)."""
    highest = max(max(max(shares), max(cutoffs)) for shares, cutoffs in map(_shares, detections))
    return max(0.3, math.ceil(highest * 1.1 * 20) / 20)


def board_details(entries: Sequence[tuple[str, int, dict]], title: str | None = None, cols: int = 2):
    """A grid of board_detail panels on one y scale. entries: (job label, board number, saved detection)."""
    rows = -(-len(entries) // cols)
    cols = min(cols, len(entries))
    fig = plt.figure(figsize=(5.8 * cols, 2.7 * rows + 0.8), layout="constrained")
    top = _bars_top([r for _, _, r in entries])
    for sub, (job, number, r) in zip(fig.subfigures(rows, cols, squeeze=False).flat, entries, strict=False):
        board_detail(sub, r, _caption(job, number, r), top)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.legend(handles=_bar_handles(), loc="outside lower center", ncol=5, fontsize=8, frameon=False)
    return fig


_KINDS = ("single_win", "double_win", "no_win")  # the browser's and the examples' order


class BoardBrowser:
    """Every saved board, one at a time, filtered by job, kind of board and outcome (needs ipywidgets).

    Display it in a notebook. Without a running kernel (on GitHub, say) only its text stands in for it.
    entries: (job label, board number, saved detection).
    """

    STATIC_TEXT = "[Board browser: run this notebook in Jupyter or VS Code to step through every board.]"

    def __init__(self, entries: Sequence[tuple[str, int, dict]]):
        import ipywidgets as widgets

        from qttt.analysis import OUTCOMES, outcome

        def order(entry):
            r = entry[2]
            return _KINDS.index(r["label"]), OUTCOMES.index(outcome(r)), entry[0], entry[1]

        self.entries = sorted(entries, key=order)
        present = lambda values: list(dict.fromkeys(values))  # noqa: E731
        self.job = widgets.Dropdown(options=["all jobs", *present(e[0] for e in self.entries)], description="job")
        self.kind = widgets.Dropdown(description="board", options=["all boards", *present(
            e[2]["label"].replace("_", " ") for e in self.entries)])
        self.outcome = widgets.Dropdown(description="outcome", options=["all outcomes", *present(
            outcome(e[2]) for e in sorted(self.entries, key=lambda e: OUTCOMES.index(outcome(e[2]))))])
        self.slider = widgets.IntSlider(min=1, max=1, description="#", continuous_update=False)
        previous, following = widgets.Button(description="◀ previous"), widgets.Button(description="next ▶")
        self.status = widgets.HTML()
        self.output = widgets.Output()
        self._outcome = outcome

        previous.on_click(lambda _: self._step(-1))
        following.on_click(lambda _: self._step(1))
        for control in (self.job, self.kind, self.outcome):
            control.observe(lambda _: self._filter(), names="value")
        self.slider.observe(lambda _: self._quiet or self._draw(), names="value")
        self._quiet = False
        self.widget = widgets.VBox([widgets.HBox([self.job, self.kind, self.outcome]),
                                    widgets.HBox([previous, self.slider, following, self.status]), self.output])
        self._filter()

    def shown(self) -> list[tuple[str, int, dict]]:
        """The entries the current filters let through, in browsing order."""
        return [e for e in self.entries
                if self.job.value in ("all jobs", e[0])
                and self.kind.value in ("all boards", e[2]["label"].replace("_", " "))
                and self.outcome.value in ("all outcomes", self._outcome(e[2]))]

    def _filter(self) -> None:
        self._quiet = True  # the slider moves twice here; draw once at the end
        self.slider.max = max(1, len(self.shown()))
        self.slider.value = 1
        self._quiet = False
        self._draw()

    def _step(self, by: int) -> None:
        self.slider.value = min(self.slider.max, max(1, self.slider.value + by))

    def _draw(self) -> None:
        from IPython.display import display

        shown = self.shown()
        self.status.value = f"board {self.slider.value} of {len(shown)}" if shown else "no board matches"
        with self.output:
            self.output.clear_output(wait=True)
            if shown:
                job, number, r = shown[self.slider.value - 1]
                fig = plt.figure(figsize=(7.5, 3.3), layout="constrained")
                board_detail(fig, r, _caption(job, number, r))
                fig.legend(handles=_bar_handles(), loc="outside lower center", ncol=3, fontsize=7, frameon=False)
                display(fig)
                plt.close(fig)

    def _repr_mimebundle_(self, **kwargs):
        return {**self.widget._repr_mimebundle_(**kwargs), "text/plain": self.STATIC_TEXT}


def cutoff_margins(detections: Sequence[dict], title: str):
    """Every line of every board as one dot: its count divided by its cutoff, grouped by kind of line.

    A winning line at or above 1 was found; below 1 it was missed. A losing line at or above 1 was flagged and
    then rejected by the check (orange x).
    """
    groups = {"winning lines\nsingle wins": [], "winning lines\ndouble wins": [],
              "losing lines\nboards with a winner": [], "losing lines\nno-win boards": []}
    for r in detections:
        for i, (count, cutoff) in enumerate(zip(r["counts"], r["cutoffs"], strict=True)):
            if i in r["true_lines"]:
                key = "winning lines\nsingle wins" if len(r["true_lines"]) == 1 else "winning lines\ndouble wins"
            else:
                key = "losing lines\nboards with a winner" if r["true_lines"] else "losing lines\nno-win boards"
            groups[key].append((count / cutoff, i in r["rejected"]))
    groups = {name: points for name, points in groups.items() if points}

    rng = np.random.default_rng(0)  # the same jitter every run
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    labels = []
    for x, (name, points) in enumerate(groups.items()):
        winning = name.startswith("winning")
        ratios = np.array([ratio for ratio, _ in points])
        flagged = np.array([rejected for _, rejected in points])
        jitter = x + rng.uniform(-0.3, 0.3, len(points))
        alpha = max(0.15, min(0.8, 30 / len(points)))
        if winning:
            ax.scatter(jitter, ratios, s=14, color=WIN, alpha=max(alpha, 0.4), linewidths=0)
            labels.append(f"{name}\nfound {int((ratios >= 1).sum())} of {len(points)}")
        else:
            ax.scatter(jitter[~flagged], ratios[~flagged], s=10, color=LOSE, alpha=alpha, linewidths=0)
            ax.scatter(jitter[flagged], ratios[flagged], s=24, color=REJECTED, marker="x")
            labels.append(f"{name}\nflagged {int(flagged.sum())} of {len(points)}")
    ax.axhline(1, color="black", linewidth=1.2)
    ax.text(len(groups) - 0.5, 1, " cutoff", va="center", ha="left", fontsize=8)
    ax.set_xticks(range(len(groups)), labels, fontsize=8)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylabel("count ÷ cutoff")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


def found_by_line(found: Sequence[int], boards: Sequence[int], title: str):
    """Share of winning lines found, by line index, with 95% Wilson intervals.

    found[i]: boards where line i was a winning line and was confirmed; boards[i]: boards where it was a winning
    line.
    """
    from qttt.stats import wilson_interval

    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    for i, (k, n) in enumerate(zip(found, boards, strict=True)):
        if not n:
            ax.text(i, 0.02, "none", ha="center", fontsize=7, color="dimgray")
            continue
        low, high = wilson_interval(k, n)
        ax.bar(i, k / n, color=WIN, alpha=0.8)
        ax.errorbar(i, k / n, yerr=[[k / n - low], [high - k / n]], color="black", capsize=3)
        ax.text(i, 1.03, f"{k}/{n}", ha="center", fontsize=7)
    ax.set_xticks(range(NUM_LINES), [f"{i}: {SHORT_LINE_NAMES[i]}" for i in range(NUM_LINES)], fontsize=7,
                  rotation=20, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("share found")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


def predicted_vs_measured(points: Mapping[str, Sequence[tuple[float, float, bool]]], title: str):
    """Share of shots on each board's winning lines, measured against each model's prediction (one dot per board).

    points: {model name: [(predicted, measured, double win)]}. On the diagonal, the model is exact.
    """
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    highest = 0.0
    for rows, color in zip(points.values(), ("tab:blue", "tab:orange", "tab:purple"), strict=False):
        for double, marker in ((False, "o"), (True, "^")):
            chosen = [(p, m) for p, m, d in rows if d == double]
            if chosen:
                x, y = zip(*chosen, strict=True)
                ax.scatter(x, y, s=22, color=color, marker=marker, alpha=0.6, linewidths=0)
                highest = max(highest, *x, *y)
    top = min(1.0, math.ceil(highest * 1.15 * 10) / 10)
    ax.plot([0, top], [0, top], color="black", linewidth=1, linestyle="--")
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_aspect("equal")
    ax.set_xlabel("predicted share on the winning lines")
    ax.set_ylabel("measured share")
    handles = [Line2D([], [], color=color, marker="o", linestyle="none", label=name)
               for name, color in zip(points, ("tab:blue", "tab:orange", "tab:purple"), strict=False)]
    handles += [Line2D([], [], color="gray", marker=m, linestyle="none", label=label)
                for m, label in (("o", "single win"), ("^", "double win"))]
    handles.append(Line2D([], [], color="black", linestyle="--", label="exact prediction"))
    ax.legend(handles=handles, fontsize=7, loc="upper left")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig
