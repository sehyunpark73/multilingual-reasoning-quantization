from dataclasses import dataclass

from matplotlib.axes import Axes
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from utils.dataset import LANGUAGES
from utils.hub import load_bf16_results, load_results
from utils.path import extract_method, extract_model_name, make_result_dir
from utils.plot import LANGUAGE_NAMES, NUM_POSITION_BINS, POSITION_LABELS, VALUE_FONTSIZE, VALUE_OFFSET, compute_figure_size, save_figure
from utils.score import find_solved_problem_indices

ERROR_TYPE_NAMES = {"fact": "사실 오류", "relation": "관계 오류", "target": "목표 오류", "doubt": "조건 무시", "arithmetic": "계산 오류", "other": "기타"}
ERROR_TYPE_COLORS = {"fact": "#2a78d6", "relation": "#eb6834", "target": "#1baf7a", "doubt": "#eda100", "arithmetic": "#e87ba4", "other": "#a3a29b"}
MIN_LABEL_RATIO = 0.05
ERROR_RATIO_YLIM = (0, 0.5)
STACK_BAR_WIDTH = 0.5
STACK_EDGE_WIDTH = 1


@dataclass
class Config:
    model_hub_path: str = "mxfp4/google/gemma-4-E2B-it"
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)
    bf16_seeds: tuple[int, ...] = (0, 1, 2)


def _compute_language_ratios(
    results: dict[int, dict],
    language: str,
    num_solved_problems: int,
) -> dict[str, float]:
    counts = dict.fromkeys(ERROR_TYPE_COLORS, 0)
    for result in results.values():
        for error_type in result[language]["error_types"]:
            counts[error_type] += 1
    return {error_type: count / (num_solved_problems * len(results)) for error_type, count in counts.items()}


def _compute_position_ratios(
    results: dict[int, dict],
    quantized_results: dict[int, dict],
    tokenizer: PreTrainedTokenizerBase,
) -> list[dict[str, float]]:
    position_counts = []
    for _ in range(NUM_POSITION_BINS):
        position_counts.append(dict.fromkeys(ERROR_TYPE_COLORS, 0))
    num_errors = 0
    for seed, result in results.items():
        for language in LANGUAGES:
            language_result = result[language]
            quantized_language_result = quantized_results[seed][language]
            for problem_index, error_line, error_type in zip(language_result["wrong_problem_indices"], language_result["error_lines"], language_result["error_types"]):
                lines = quantized_language_result["reasonings"][problem_index].split("\n")
                error_token_index = len(tokenizer("\n".join(lines[: error_line - 1]), add_special_tokens=False)["input_ids"])
                position_index = min(error_token_index * NUM_POSITION_BINS // quantized_language_result["reasoning_lens"][problem_index], NUM_POSITION_BINS - 1)
                position_counts[position_index][error_type] += 1
                num_errors += 1
    position_ratios = []
    for counts in position_counts:
        position_ratios.append({error_type: count / num_errors for error_type, count in counts.items()})
    return position_ratios


def _plot_error_type_bars(
    ax: Axes,
    groups: list[str],
    ratios: list[dict[str, float]],
) -> None:
    for x, group_ratios in enumerate(ratios):
        bottom = 0
        for error_type, color in ERROR_TYPE_COLORS.items():
            ratio = group_ratios[error_type]
            ax.bar(x, ratio, bottom=bottom, width=STACK_BAR_WIDTH, color=color, edgecolor="white", linewidth=STACK_EDGE_WIDTH)
            if ratio >= ERROR_RATIO_YLIM[1] * MIN_LABEL_RATIO:
                ax.text(x, bottom + ratio / 2, f"{ratio:.1%}", ha="center", va="center", fontsize=VALUE_FONTSIZE)
            bottom += ratio
        ax.annotate(f"{bottom:.1%}", (x, bottom), xytext=(0, VALUE_OFFSET), textcoords="offset points", ha="center", va="bottom", fontsize=VALUE_FONTSIZE)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylim(ERROR_RATIO_YLIM)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))


def main() -> None:
    config = parse(Config)
    model_name = extract_model_name(config.model_hub_path)
    method = extract_method(config.model_hub_path)
    results = load_results(make_result_dir("error_types", "mgsm"), config.model_hub_path, config.max_generation_len, config.seeds)
    quantized_results = load_results(make_result_dir("loss", "mgsm"), config.model_hub_path, config.max_generation_len, config.seeds)
    bf16_results = load_bf16_results("mgsm", model_name, config.max_generation_len, config.bf16_seeds)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    language_labels = []
    language_ratios = []
    for language in LANGUAGES:
        num_solved_problems = len(find_solved_problem_indices(bf16_results, language))
        language_labels.append(f"{LANGUAGE_NAMES[language]}\n{num_solved_problems}문제")
        language_ratios.append(_compute_language_ratios(results, language, num_solved_problems))
    position_ratios = _compute_position_ratios(results, quantized_results, tokenizer)

    fig, axes = plt.subplots(1, 2, figsize=compute_figure_size(1), layout="constrained")
    _plot_error_type_bars(axes[0], language_labels, language_ratios)
    axes[0].set_title(method)
    axes[0].set_ylabel("오류 비율")
    _plot_error_type_bars(axes[1], POSITION_LABELS, position_ratios)
    axes[1].set_title(f"{method} 모든 언어")
    axes[1].set_ylabel("첫 오류 위치")

    fig.suptitle("오류 유형")
    handles = [Patch(color=color, label=ERROR_TYPE_NAMES[error_type]) for error_type, color in ERROR_TYPE_COLORS.items()]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    save_figure(fig, model_name, "error_types")


if __name__ == "__main__":
    main()
