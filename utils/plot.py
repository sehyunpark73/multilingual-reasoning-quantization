from pathlib import Path
from statistics import mean

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator, PercentFormatter

from utils.dataset import LANGUAGES

LANGUAGE_NAMES = {"en": "영어", "zh": "중국어", "ja": "일본어", "sw": "스와힐리어", "te": "텔루구어"}
LANGUAGE_COLORS = {"en": "#3c5a9a", "zh": "#de2910", "ja": "#f4a6b7", "sw": "#1eb53a", "te": "#ff9933"}
RESOURCE_LANGUAGES = {"고자원": ["en"], "중자원": ["zh", "ja"], "저자원": ["sw", "te"]}
GROUP_GAP = 1
GROUP_WIDTH = len(LANGUAGES) + GROUP_GAP
VALUE_FONTSIZE = 8
VALUE_OFFSET = 3
VALUE_MARGIN = 0.15
BASELINE_COLOR = "#808080"
BASELINE_WIDTH = 0.8
TICK_STEPS = [1, 2, 5, 10]
NO_INSERT_ALPHA = 0.35
INSERT_LOSS_YLIM = (-0.05, 0.2)
INSERT_LOSS_YTICK = 0.05
NUM_POSITION_BINS = 5
POSITION_LABELS = ["0~20%", "20~40%", "40~60%", "60~80%", "80~100%"]

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


def _format_percent_point(
    value: float,
) -> str:
    percent_point = round(value * 100, 1)
    if percent_point == 0:
        percent_point = 0.0
    return f"{percent_point:.1f}%p"


def _compute_bar_x(
    group_index: int,
    language_index: int,
) -> float:
    return group_index * GROUP_WIDTH + language_index - (len(LANGUAGES) - 1) / 2


def plot_loss_bars(
    ax: Axes,
    groups: list[str],
    losses: dict[str, list[float]],
) -> None:
    for language_index, language in enumerate(LANGUAGES):
        for group_index, loss in enumerate(losses[language]):
            x = _compute_bar_x(group_index, language_index)
            ax.bar(x, loss, width=1, color=LANGUAGE_COLORS[language])
            ax.annotate(_format_percent_point(loss), (x, max(loss, 0)), xytext=(0, VALUE_OFFSET), textcoords="offset points", ha="center", va="bottom", fontsize=VALUE_FONTSIZE)
    ax.axhline(0, color=BASELINE_COLOR, linewidth=BASELINE_WIDTH)
    ax.set_xticks([group_index * GROUP_WIDTH for group_index in range(len(groups))], groups)
    ax.yaxis.set_major_locator(MaxNLocator(steps=TICK_STEPS))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0, symbol="%p"))
    ax.margins(y=VALUE_MARGIN)


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
            text = _format_percent_point(insert_loss)
            if groups[group_index] in recovery_groups:
                text = f"{text}\n{(no_insert_loss - insert_loss) / no_insert_loss:.0%} 회복"
            y = max(no_insert_loss, insert_loss, 0)
            offset = VALUE_OFFSET
            vertical_alignment = "bottom"
            if insert_loss < 0:
                y = insert_loss
                offset = -VALUE_OFFSET
                vertical_alignment = "top"
            ax.annotate(text, (x, y), xytext=(0, offset), textcoords="offset points", ha="center", va=vertical_alignment, fontsize=VALUE_FONTSIZE)
    ax.axhline(0, color=BASELINE_COLOR, linewidth=BASELINE_WIDTH)
    ax.set_xticks([group_index * GROUP_WIDTH for group_index in range(len(groups))], groups)
    ax.yaxis.set_major_locator(MultipleLocator(INSERT_LOSS_YTICK))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0, symbol="%p"))


def make_insert_handles() -> tuple[Patch, ...]:
    return Patch(visible=False, label=""), Patch(color=BASELINE_COLOR, alpha=NO_INSERT_ALPHA, label="삽입 없음"), Patch(color=BASELINE_COLOR, label="삽입")


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


def save_figure(
    fig: Figure,
    model_name: str,
    figure_name: str,
) -> None:
    path = Path("figures") / model_name.split("/")[1] / f"{figure_name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"saved {path}")
