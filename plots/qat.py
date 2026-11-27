from dataclasses import dataclass

import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.hub import load_bf16_results, load_results
from utils.path import extract_model_name, make_result_dir
from utils.plot import add_language_legend, compute_figure_size, make_solved_count_labels, plot_language_bars, save_figure
from utils.score import compute_accuracy_losses, compute_subset_accuracy_losses, split_solved_problem_indices

MODEL_NAMES = ["mxfp4", "ce_kl finetune", "ce_kl finetune\nmxfp4", "kl qat", "ce_kl qat"]


@dataclass
class Config:
    model_hub_paths: tuple[str, ...] = (
        "mxfp4/google/gemma-4-E2B-it",
        "question_finetune/google/gemma-4-E2B-it/loss=ce_kl,num_samples=20000,num_trained_samples=20000",
        "question_finetune_mxfp4/google/gemma-4-E2B-it/loss=ce_kl,num_samples=20000,num_trained_samples=20000",
        "qat/google/gemma-4-E2B-it/loss=kl,num_samples=20000,num_trained_samples=20000",
        "qat/google/gemma-4-E2B-it/loss=ce_kl,num_samples=20000,num_trained_samples=20000",
    )
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def main() -> None:
    config = parse(Config)
    model_name = extract_model_name(config.model_hub_paths[0])
    bf16_results = load_bf16_results("mgsm", model_name, config.max_generation_len, config.seeds)
    all_results = [load_results(make_result_dir("loss", "mgsm"), model_hub_path, config.max_generation_len, config.seeds) for model_hub_path in config.model_hub_paths]
    solved_problem_indices_by_language, unsolved_problem_indices_by_language = split_solved_problem_indices(bf16_results)

    fig, axes = plt.subplots(3, 1, figsize=compute_figure_size(3), layout="constrained")
    plot_language_bars(axes[0], MODEL_NAMES, compute_accuracy_losses(bf16_results, all_results), "%p")
    plot_language_bars(axes[1], MODEL_NAMES, compute_subset_accuracy_losses(bf16_results, all_results, solved_problem_indices_by_language), "%p")
    plot_language_bars(axes[2], MODEL_NAMES, compute_subset_accuracy_losses(bf16_results, all_results, unsolved_problem_indices_by_language), "%p")
    for ax, ylabel in zip(axes, ["손실", "bf16이 푼 문제\n손실", "bf16이 못 푼 문제\n손실"]):
        ax.set_ylabel(ylabel)

    fig.suptitle("qat")
    add_language_legend(fig, make_solved_count_labels(solved_problem_indices_by_language, unsolved_problem_indices_by_language))
    save_figure(fig, model_name, "qat")


if __name__ == "__main__":
    main()
