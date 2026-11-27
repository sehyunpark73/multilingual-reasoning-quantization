from dataclasses import dataclass

import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.dataset import Benchmark
from utils.hub import load_bf16_results, load_results
from utils.path import extract_method, extract_model_name, make_result_dir
from utils.plot import LANGUAGE_NAMES, add_language_legend, compute_figure_size, plot_language_bars, save_figure
from utils.score import compute_accuracy_losses, compute_loop_ratio, compute_mean_values, compute_truncated_ratio

SEEDS = {"mgsm": (0, 1, 2), "mmlu_prox_lite": (0,)}
BF16_SEEDS = {"mgsm": (0, 1, 2), "mmlu_prox_lite": (0, 1)}


@dataclass
class Config:
    model_hub_paths: tuple[str, ...] = (
        "gptq/google/gemma-4-E2B-it/calibration_languages=en,num_bits=4,group_size=128,num_chunks=128",
        "awq/google/gemma-4-E2B-it/calibration_languages=en,num_bits=4,group_size=128,num_chunks=128",
        "nf4/google/gemma-4-E2B-it",
        "mxfp4/google/gemma-4-E2B-it",
    )
    benchmark: Benchmark = "mgsm"
    max_generation_len: int = 16384


def main() -> None:
    config = parse(Config)
    model_name = extract_model_name(config.model_hub_paths[0])
    methods = [extract_method(model_hub_path) for model_hub_path in config.model_hub_paths]

    bf16_results = load_bf16_results(config.benchmark, model_name, config.max_generation_len, BF16_SEEDS[config.benchmark])
    all_results = [load_results(make_result_dir("loss", config.benchmark), model_hub_path, config.max_generation_len, SEEDS[config.benchmark]) for model_hub_path in config.model_hub_paths]

    fig, axes = plt.subplots(3, 1, figsize=compute_figure_size(3), layout="constrained")
    plot_language_bars(axes[0], methods, compute_accuracy_losses(bf16_results, all_results), "%p")
    plot_language_bars(axes[1], methods, compute_mean_values(all_results, compute_truncated_ratio), "%")
    plot_language_bars(axes[2], methods, compute_mean_values(all_results, compute_loop_ratio), "%")
    for ax, ylabel in zip(axes, ["손실", "잘림", "반복"]):
        ax.set_ylabel(ylabel)
    axes[0].set_title(config.benchmark)

    fig.suptitle("언어별 손실")
    add_language_legend(fig, LANGUAGE_NAMES)
    save_figure(fig, model_name, f"loss_{config.benchmark}")


if __name__ == "__main__":
    main()
