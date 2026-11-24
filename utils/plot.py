from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter

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


def extract_method(
    model_hub_path: str,
) -> str:
    return model_hub_path.split("/")[0]


def _format_percent_point(
    value: float,
) -> str:
    return f"{round(value * 100, 1) + 0.0:.1f}%p"


def plot_loss_bars(
    ax: Axes,
    groups: list[str],
    losses: dict[str, list[float]],
) -> None:
    for language_index, language in enumerate(LANGUAGES):
        xs = [group_index * GROUP_WIDTH + language_index - (len(LANGUAGES) - 1) / 2 for group_index in range(len(groups))]
        ax.bar(xs, losses[language], width=1, color=LANGUAGE_COLORS[language])
        for x, loss in zip(xs, losses[language]):
            ax.annotate(_format_percent_point(loss), (x, max(loss, 0)), xytext=(0, VALUE_OFFSET), textcoords="offset points", ha="center", va="bottom", fontsize=VALUE_FONTSIZE)
    ax.axhline(0, color=BASELINE_COLOR, linewidth=BASELINE_WIDTH)
    ax.set_xticks([group_index * GROUP_WIDTH for group_index in range(len(groups))], groups)
    ax.yaxis.set_major_locator(MaxNLocator(steps=TICK_STEPS))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0, symbol="%p"))
    ax.margins(y=VALUE_MARGIN)


def add_language_legend(
    fig: Figure,
    labels: dict[str, str],
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
    fig.legend(handles=handles, loc="outside lower center", ncol=len(RESOURCE_LANGUAGES))


def save_figure(
    fig: Figure,
    model_name: str,
    figure_name: str,
) -> None:
    path = Path("figures") / model_name.split("/")[1] / f"{figure_name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"saved {path}")
