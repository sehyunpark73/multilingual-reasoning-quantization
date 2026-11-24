from dataclasses import dataclass

import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.dataset import LANGUAGES, Benchmark
from utils.hub import load_bf16_results, load_results
from utils.path import extract_model_name, make_result_dir
from utils.plot import LANGUAGE_NAMES, add_language_legend, extract_method, plot_loss_bars, save_figure
from utils.score import compute_mean_accuracy

BENCHMARKS: list[Benchmark] = ["mgsm", "mmlu_prox_lite"]
SEEDS = {"mgsm": (0, 1, 2), "mmlu_prox_lite": (0,)}
BF16_SEEDS = {"mgsm": (0, 1, 2), "mmlu_prox_lite": (0, 1)}
FIGURE_SIZE = (10, 6)


@dataclass
class Config:
    model_hub_paths: tuple[str, ...] = (
        "gptq/google/gemma-4-E2B-it/calibration_languages=en,num_bits=4,group_size=128,num_chunks=128",
        "awq/google/gemma-4-E2B-it/calibration_languages=en,num_bits=4,group_size=128,num_chunks=128",
        "nf4/google/gemma-4-E2B-it",
        "mxfp4/google/gemma-4-E2B-it",
    )
    max_generation_len: int = 16384


def _compute_losses(
    model_name: str,
    model_hub_paths: tuple[str, ...],
    benchmark: Benchmark,
    max_generation_len: int,
) -> dict[str, list[float]]:
    bf16_results = load_bf16_results(benchmark, model_name, max_generation_len, BF16_SEEDS[benchmark])
    losses = {language: [] for language in LANGUAGES}
    for model_hub_path in model_hub_paths:
        results = load_results(make_result_dir("loss", benchmark), model_hub_path, max_generation_len, SEEDS[benchmark])
        for language in LANGUAGES:
            losses[language].append(compute_mean_accuracy(bf16_results, language) - compute_mean_accuracy(results, language))
    return losses


def main() -> None:
    config = parse(Config)
    model_name = extract_model_name(config.model_hub_paths[0])
    methods = [extract_method(model_hub_path) for model_hub_path in config.model_hub_paths]

    fig, axes = plt.subplots(len(BENCHMARKS), 1, figsize=FIGURE_SIZE, layout="constrained")
    for ax, benchmark in zip(axes, BENCHMARKS):
        losses = _compute_losses(model_name, config.model_hub_paths, benchmark, config.max_generation_len)
        plot_loss_bars(ax, methods, losses)
        ax.set_title(benchmark)
        ax.set_ylabel("손실")

    fig.suptitle("언어별 손실")
    add_language_legend(fig, LANGUAGE_NAMES)
    save_figure(fig, model_name, "loss")


if __name__ == "__main__":
    main()
