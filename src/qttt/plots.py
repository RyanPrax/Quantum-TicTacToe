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
                 winners=(), rejected=(), title: str | None = None) -> None:
    """A board with the circuit's output drawn on it: every line through its cells, as thick as its share of shots.

    Green: a real winning line. Orange, dashed: flagged (reached its cutoff) and rejected by the check. Gray: a
    losing line. Each line's index and share are written where it leaves the board, in bold when it reached its
    cutoff.
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
        ax.text(*_SHARE_LABELS[i], f"{i}: {share:.0%}", ha="center", va="center", fontsize=7,
                fontweight="bold" if flagged else "normal", color=color if color != LOSE else "dimgray")
    if title:
        ax.set_title(title, fontsize=7)


def board_results(panels: Sequence[tuple[Sequence[str], Sequence[int], Sequence[int] | None, Sequence[int],
                                         Sequence[int], str]], title: str | None = None, cols: int = 4):
    """A grid of board_result panels: (board, counts, cutoffs, winning lines, rejected lines, panel title)."""
    rows = -(-len(panels) // cols)
    cols = min(cols, len(panels))
    fig, axes = plt.subplots(rows, cols, figsize=(2.9 * cols, 3.0 * rows + 0.4), squeeze=False)
    for ax, (board, counts, cutoffs, winners, rejected, panel) in zip(axes.flat, panels, strict=False):
        board_result(ax, board, counts, cutoffs, winners, rejected, panel)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    handles = [Line2D([], [], color=WIN, linewidth=5, alpha=0.6, label="winning line"),
               Line2D([], [], color=LOSE, linewidth=5, alpha=0.45, label="losing line"),
               Line2D([], [], color=REJECTED, linewidth=3, linestyle="--", label="flagged, rejected"),
               Line2D([], [], color="none", label="thickness: share of shots; bold %: reached cutoff")]
    fig.legend(handles=handles, loc="lower center", ncol=2 if cols < 3 else 4, fontsize=7, frameon=False)
    height = fig.get_size_inches()[1]
    fig.tight_layout(rect=(0, 0.35 / height * (2 if cols < 3 else 1), 1, 1 - (0.3 / height if title else 0)))
    return fig


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
    handles = [Patch(color=WIN, label="winning line"), Patch(color=LOSE, label="losing line")]
    if rejected:
        handles.append(Patch(facecolor=REJECTED, hatch=REJECTED_HATCH, edgecolor="white",
                             label="x: flagged, rejected by the check"))
    if cutoffs:
        handles.append(Line2D([], [], color="black", linewidth=1.5, label="cutoff"))
    if even:
        handles.append(Line2D([], [], color="gray", linestyle=":", label="1/8 (even spread)"))
    width, height = fig.get_size_inches()
    ncol = len(handles) if width >= 11 else 3
    fig.legend(handles=handles, loc="lower center", ncol=ncol, fontsize=8, frameon=False)
    return 0.3 * -(-len(handles) // ncol) / height


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
    fig, ax = plt.subplots(figsize=(6, 3))
    for m, style in ((1, "o-"), (2, "s--")):
        ax.plot(ks, np.sin((2 * ks + 1) * np.arcsin(np.sqrt(m / 8))) ** 2, style, label=f"M = {m} winning line(s)")
    ax.axhline(2 / 8, color="gray", linestyle=":", linewidth=1, label="M = 2 by chance (2 of 8)")
    ax.axvline(1, color=WIN, alpha=0.25, linewidth=8)
    ax.set_xticks(ks)
    ax.set_xlabel("Grover iterations")
    ax.set_ylabel("probability on winning lines")
    ax.set_title("1 iteration is the only count that separates single, double and no win", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def grover_rotation():
    """The state as a vector in the (|bad>, |good>) plane after k = 0, 1, 2 iterations, M = 1 and M = 2."""
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="lightgray"))
    for m, color in ((1, "tab:blue"), (2, "tab:orange")):
        theta = math.asin(math.sqrt(m / 8))
        for k, alpha in ((0, 0.35), (1, 1.0), (2, 0.35)):
            angle = (2 * k + 1) * theta
            ax.annotate("", xy=(math.cos(angle), math.sin(angle)), xytext=(0, 0),
                        arrowprops={"arrowstyle": "->", "color": color, "alpha": alpha, "linewidth": 2})
            ax.text(1.08 * math.cos(angle), 1.08 * math.sin(angle), f"M={m}, k={k}", color=color, alpha=alpha,
                    fontsize=7, ha="center", va="center")
    ax.set_xlim(-1.25, 1.25)
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


def readout_matrices(profiles: Mapping[str, object]):
    """One readout matrix A per job. profiles: {job label: NoiseProfile}."""
    fig, axes = plt.subplots(1, len(profiles), figsize=(3.4 * len(profiles), 3.2), squeeze=False)
    image = None
    for ax, (label, profile) in zip(axes[0], profiles.items(), strict=True):
        image = ax.imshow(profile.readout, cmap="viridis", vmin=0, vmax=1)
        ax.set_xlabel("prepared line index")
        ax.set_ylabel("measured line index")
        ax.set_title(f"Readout matrix A\n{label}", fontsize=9)
    fig.colorbar(image, ax=axes[0].tolist(), fraction=0.046)
    return fig


def board_bars(detections: Sequence[tuple[int, dict, str]], title: str, cols: int = 5):
    """Every board's counts. detections: (board number, saved detection, panel title)."""
    rows = -(-len(detections) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.4 * rows), sharey=True, squeeze=False)
    for ax, (_, r, panel) in zip(axes.flat, detections, strict=False):
        shots = r["shots"]
        line_bars(ax, [c / shots for c in r["counts"]], r["true_lines"], r["rejected"],
                  [k / shots for k in r["cutoffs"]], even=1 / 8)
        ax.set_title(panel, fontsize=6, wrap=True)
        ax.set_ylim(0, 1)
    for ax in axes.flat[len(detections):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, line_legend(fig), 1, 1))
    return fig


def measured_vs_predicted(numbered: Sequence[tuple[int, dict]], models: Mapping[str, object], title: str):
    """Share of shots on each board's winning lines: measured, and predicted by each noise profile."""
    wins = [r for _, r in numbered]
    width = 0.8 / (len(models) + 1)
    x = np.arange(len(wins))
    measured = [sum(r["counts"][i] for i in r["true_lines"]) / r["shots"] for r in wins]
    fig, ax = plt.subplots(figsize=(0.45 * len(wins) + 2.5, 3))
    ax.bar(x, measured, width, label="hardware result", color=WIN)
    for k, ((name, profile), hatch) in enumerate(zip(models.items(), ("", "..", "xx"), strict=False), start=1):
        predicted = [sum(profile.predicted_for(r["true_lines"])[i] for i in r["true_lines"]) for r in wins]
        ax.bar(x + k * width, predicted, width, label=name, alpha=0.7, hatch=hatch)
    ax.axhline(1 / 8, color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(x + width * len(models) / 2, [f"#{number}" for number, _ in numbered], fontsize=7)
    ax.set_xlabel("board (numbers as in the table above)")
    ax.set_ylabel("share of shots on winning lines")
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig
