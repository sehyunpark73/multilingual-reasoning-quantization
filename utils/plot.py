from pathlib import Path
from statistics import mean
from typing import Literal, TypeAlias

from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter, StrMethodFormatter

from utils.dataset import LANGUAGES

LANGUAGE_NAMES = {"en": "영어", "zh": "중국어", "ja": "일본어", "sw": "스와힐리어", "te": "텔루구어"}
LANGUAGE_COLORS = {"en": "#3c5a9a", "zh": "#de2910", "ja": "#f4a6b7", "sw": "#1eb53a", "te": "#ff9933"}
RESOURCE_LANGUAGES = {"고자원": ["en"], "중자원": ["zh", "ja"], "저자원": ["sw", "te"]}
GROUP_GAP = 1
GROUP_WIDTH = len(LANGUAGES) + GROUP_GAP
VALUE_FONTSIZE = 8
VALUE_OFFSET = 3
BASELINE_COLOR = "#808080"
BASELINE_WIDTH = 0.8
NO_INSERT_ALPHA = 0.35
YLIMS = {"%p": (-0.2, 0.2), "%": (0, 1), "배": (0, 2)}
YTICKS = {"%p": 0.1, "%": 0.25, "배": 0.5}
NUM_POSITION_BINS = 5
POSITION_LABELS = ["0~20%", "20~40%", "40~60%", "60~80%", "80~100%"]
PANEL_HEIGHT = 1.9
FIGURE_HEIGHT_MARGIN = 1.6
NUM_FIT_STEPS = 3
BAR_WIDTH = 0.45
INITIAL_FIGURE_WIDTH = 10

Unit: TypeAlias = Literal["%p", "%", "배"]

plt.rcParams.update({
    "font.family": "Apple SD Gothic Neo",
    "axes.unicode_minus": False,
    "figure.titlesize": 13,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": "#e8e7e3",
    "grid.linewidth": 0.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "savefig.dpi": 200,
})


def _format_value(
    value: float,
    unit: Unit,
) -> str:
    if unit in ("%p", "%"):
        percent = round(value * 100, 1)
        if percent == 0:
            percent = 0.0
        return f"{percent:.1f}{unit}"
    if unit == "배":
        return f"{value:.2f}배"


def _compute_bar_x(
    group_index: int,
    language_index: int,
) -> float:
    return group_index * GROUP_WIDTH + language_index - (len(LANGUAGES) - 1) / 2


def _annotate_value(
    ax: Axes,
    x: float,
    y: float,
    text: str,
    unit: Unit,
) -> None:
    ymin, ymax = YLIMS[unit]
    offset = VALUE_OFFSET
    vertical_alignment = "bottom"
    if y < 0:
        offset = -VALUE_OFFSET
        vertical_alignment = "top"
    if y > ymax:
        y = ymax
        offset = -VALUE_OFFSET
        vertical_alignment = "top"
    if y < ymin:
        y = ymin
        offset = VALUE_OFFSET
        vertical_alignment = "bottom"
    ax.annotate(text, (x, y), xytext=(0, offset), textcoords="offset points", ha="center", va=vertical_alignment, fontsize=VALUE_FONTSIZE)


def _set_yaxis(
    ax: Axes,
    unit: Unit,
) -> None:
    ax.set_ylim(YLIMS[unit])
    ax.yaxis.set_major_locator(MultipleLocator(YTICKS[unit]))
    if unit in ("%p", "%"):
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0, symbol=unit))
    if unit == "배":
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.1f}배"))


def plot_language_bars(
    ax: Axes,
    groups: list[str],
    values: dict[str, list[float]],
    unit: Unit,
) -> None:
    for language_index, language in enumerate(LANGUAGES):
        for group_index, value in enumerate(values[language]):
            x = _compute_bar_x(group_index, language_index)
            ax.bar(x, value, width=1, color=LANGUAGE_COLORS[language])
            _annotate_value(ax, x, value, _format_value(value, unit), unit)
    baseline = 0
    if unit == "배":
        baseline = 1
    ax.axhline(baseline, color=BASELINE_COLOR, linewidth=BASELINE_WIDTH)
    ax.set_xticks([group_index * GROUP_WIDTH for group_index in range(len(groups))], groups)
    _set_yaxis(ax, unit)


def plot_insert_bars(
    ax: Axes,
    groups: list[str],
    losses: dict[str, list[tuple[float, float]]],
    recovery_groups: list[str],
) -> None:
    for language_index, language in enumerate(LANGUAGES):
        for group_index, (no_insert_loss, insert_loss) in enumerate(losses[language]):
            x = _compute_bar_x(group_index, language_index)
            ax.bar(x, no_insert_loss, width=1, color=LANGUAGE_COLORS[language], alpha=NO_INSERT_ALPHA)
            ax.bar(x, insert_loss, width=1, color=LANGUAGE_COLORS[language])
            text = _format_value(insert_loss, "%p")
            if groups[group_index] in recovery_groups:
                text = f"{text}\n{(no_insert_loss - insert_loss) / no_insert_loss:.0%} 회복"
            y = max(no_insert_loss, insert_loss, 0)
            if insert_loss < 0:
                y = insert_loss
            _annotate_value(ax, x, y, text, "%p")
    ax.axhline(0, color=BASELINE_COLOR, linewidth=BASELINE_WIDTH)
    ax.set_xticks([group_index * GROUP_WIDTH for group_index in range(len(groups))], groups)
    _set_yaxis(ax, "%p")


def make_insert_handles(
    no_insert_label: str,
    insert_label: str,
) -> tuple[Patch, ...]:
    return Patch(visible=False, label=""), Patch(color=BASELINE_COLOR, alpha=NO_INSERT_ALPHA, label=no_insert_label), Patch(color=BASELINE_COLOR, label=insert_label)


def add_language_legend(
    fig: Figure,
    labels: dict[str, str],
    extra_handles: tuple[Patch, ...] = (),
) -> None:
    max_num_languages = 0
    for languages in RESOURCE_LANGUAGES.values():
        max_num_languages = max(max_num_languages, len(languages))
    handles = []
    for resource, languages in RESOURCE_LANGUAGES.items():
        handles.append(Patch(visible=False, label=resource))
        for language in languages:
            handles.append(Patch(color=LANGUAGE_COLORS[language], label=labels[language]))
        for _ in range(max_num_languages - len(languages)):
            handles.append(Patch(visible=False, label=""))
    handles += extra_handles
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles) // (max_num_languages + 1))


def make_problem_count_labels(
    problem_indices_by_language: dict[str, dict[int, set[int]]],
) -> dict[str, str]:
    labels = {}
    for language, problem_indices_by_seed in problem_indices_by_language.items():
        num_problems = round(mean(len(problem_indices) for problem_indices in problem_indices_by_seed.values()))
        labels[language] = f"{LANGUAGE_NAMES[language]} {num_problems}문제"
    return labels


def compute_figure_size(
    num_rows: int,
    width: float = INITIAL_FIGURE_WIDTH,
) -> tuple[float, float]:
    return width, num_rows * PANEL_HEIGHT + FIGURE_HEIGHT_MARGIN


def _fit_panel_height(
    fig: Figure,
) -> None:
    num_rows = fig.axes[0].get_subplotspec().get_gridspec().nrows
    for _ in range(NUM_FIT_STEPS):
        fig.canvas.draw()
        panel_height = fig.axes[0].get_window_extent().height / fig.dpi
        width, height = fig.get_size_inches()
        fig.set_size_inches(width, height + (PANEL_HEIGHT - panel_height) * num_rows)


def _fit_bar_width(
    fig: Figure,
) -> None:
    bar_axes = []
    for ax in fig.axes:
        if ax.containers and isinstance(ax.containers[0], BarContainer):
            bar_axes.append(ax)
    if not bar_axes:
        return
    widest_ax = bar_axes[0]
    for ax in bar_axes:
        if ax.get_xlim()[1] - ax.get_xlim()[0] > widest_ax.get_xlim()[1] - widest_ax.get_xlim()[0]:
            widest_ax = ax
    widest_xmin, widest_xmax = widest_ax.get_xlim()
    xlim_len = widest_xmax - widest_xmin
    for ax in bar_axes:
        offset = widest_ax.get_xticks()[-1] - ax.get_xticks()[-1]
        ax.set_xlim(widest_xmin - offset, widest_xmax - offset)
    bar_data_width = bar_axes[0].containers[0].patches[0].get_width()
    num_columns = bar_axes[0].get_subplotspec().get_gridspec().ncols
    for _ in range(NUM_FIT_STEPS):
        fig.canvas.draw()
        bar_width = bar_data_width * bar_axes[0].get_window_extent().width / fig.dpi / xlim_len
        width, height = fig.get_size_inches()
        fig.set_size_inches(width + (BAR_WIDTH - bar_width) * xlim_len / bar_data_width * num_columns, height)


def save_figure(
    fig: Figure,
    model_name: str,
    figure_name: str,
) -> None:
    _fit_bar_width(fig)
    _fit_panel_height(fig)
    path = Path("figures") / model_name.split("/")[1] / f"{figure_name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"saved {path}")
