from dataclasses import dataclass
from statistics import median

import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.hub import load_bf16_results, load_results
from utils.path import extract_model_name, make_result_dir
from utils.plot import LANGUAGE_NAMES, add_language_legend, compute_figure_size, plot_language_bars, save_figure
from utils.score import compute_accuracy_losses, compute_loop_ratio, compute_mean_values, compute_truncated_ratio

FINETUNE_NAMES = ["ce\n800 sample", "ce_kl\n800 sample", "ce_kl\n4000 sample", "ce_kl\n20000 sample", "ce_kl\n29492 sample"]


@dataclass
class Config:
    model_hub_paths: tuple[str, ...] = (
        "question_finetune/google/gemma-4-E2B-it/loss=ce,num_samples=20000,num_trained_samples=800",
        "question_finetune/google/gemma-4-E2B-it/loss=ce_kl,num_samples=20000,num_trained_samples=800",
        "question_finetune/google/gemma-4-E2B-it/loss=ce_kl,num_samples=20000,num_trained_samples=4000",
        "question_finetune/google/gemma-4-E2B-it/loss=ce_kl,num_samples=20000,num_trained_samples=20000",
        "question_finetune/google/gemma-4-E2B-it/loss=ce_kl,num_samples=29492,num_trained_samples=29492",
    )
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def _compute_median_reasoning_len(
    language_result: dict,
) -> float:
    return median(language_result["reasoning_lens"])


def main() -> None:
    config = parse(Config)
    model_name = extract_model_name(config.model_hub_paths[0])
    bf16_results = load_bf16_results("mgsm", model_name, config.max_generation_len, config.seeds)
    all_results = [load_results(make_result_dir("loss", "mgsm"), model_hub_path, config.max_generation_len, config.seeds) for model_hub_path in config.model_hub_paths]
    median_reasoning_lens = compute_mean_values([bf16_results] + all_results, _compute_median_reasoning_len)
    reasoning_len_ratios = {}
    for language, language_median_reasoning_lens in median_reasoning_lens.items():
        reasoning_len_ratios[language] = [median_reasoning_len / language_median_reasoning_lens[0] for median_reasoning_len in language_median_reasoning_lens[1:]]

    fig, axes = plt.subplots(4, 1, figsize=compute_figure_size(4), layout="constrained")
    plot_language_bars(axes[0], FINETUNE_NAMES, compute_accuracy_losses(bf16_results, all_results), "%p")
    plot_language_bars(axes[1], FINETUNE_NAMES, compute_mean_values(all_results, compute_truncated_ratio), "%")
    plot_language_bars(axes[2], FINETUNE_NAMES, compute_mean_values(all_results, compute_loop_ratio), "%")
    plot_language_bars(axes[3], FINETUNE_NAMES, reasoning_len_ratios, "배")
    for ax, ylabel in zip(axes, ["손실", "잘림", "반복", "reasoning 길이"]):
        ax.set_ylabel(ylabel)

    fig.suptitle("영어 question finetune")
    add_language_legend(fig, LANGUAGE_NAMES)
    save_figure(fig, model_name, "question_finetune")


if __name__ == "__main__":
    main()
