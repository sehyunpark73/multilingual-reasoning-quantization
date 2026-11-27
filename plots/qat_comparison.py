from dataclasses import dataclass

import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.dataset import Benchmark
from utils.hub import load_bf16_results, load_results
from utils.path import extract_method, extract_model_name, make_result_dir
from utils.plot import add_language_legend, compute_figure_size, make_solved_count_labels, plot_language_bars, save_figure
from utils.score import compute_accuracy_losses, compute_subset_accuracy_losses, split_solved_problem_indices


@dataclass
class Config:
    model_hub_paths: tuple[str, ...] = (
        "nvfp4/Qwen/Qwen3.5-2B",
        "qat_comparison/Qwen/Qwen3.5-2B/method=nvfp4,loss=ce_kl,num_samples=20000,num_trained_samples=20000",
        "qat_comparison/Qwen/Qwen3.5-2B/method=nvfp4,loss=qad,num_samples=20000,num_trained_samples=20000",
    )
    benchmark: Benchmark = "mgsm"
    max_generation_len: int = 32768
    seeds: tuple[int, ...] = (0, 1, 2)


def _make_group_name(
    model_hub_path: str,
) -> str:
    path_parts = model_hub_path.split("/")
    if len(path_parts) == 3:
        return extract_method(model_hub_path)
    settings = path_parts[3].split(",")
    method = ""
    loss = ""
    for setting in settings:
        if setting.startswith("method="):
            method = setting.removeprefix("method=")
        if setting.startswith("loss="):
            loss = setting.removeprefix("loss=")
    return f"{loss} qat\n{method}"


def main() -> None:
    config = parse(Config)
    model_name = extract_model_name(config.model_hub_paths[0])
    group_names = [_make_group_name(model_hub_path) for model_hub_path in config.model_hub_paths]
    bf16_results = load_bf16_results(config.benchmark, model_name, config.max_generation_len, config.seeds)
    all_results = [load_results(make_result_dir("loss", config.benchmark), model_hub_path, config.max_generation_len, config.seeds) for model_hub_path in config.model_hub_paths]
    solved_problem_indices_by_language, unsolved_problem_indices_by_language = split_solved_problem_indices(bf16_results)

    fig, axes = plt.subplots(3, 1, figsize=compute_figure_size(3), layout="constrained")
    plot_language_bars(axes[0], group_names, compute_accuracy_losses(bf16_results, all_results), "%p")
    plot_language_bars(axes[1], group_names, compute_subset_accuracy_losses(bf16_results, all_results, solved_problem_indices_by_language), "%p")
    plot_language_bars(axes[2], group_names, compute_subset_accuracy_losses(bf16_results, all_results, unsolved_problem_indices_by_language), "%p")
    for ax, ylabel in zip(axes, ["손실", "bf16이 푼 문제\n손실", "bf16이 못 푼 문제\n손실"]):
        ax.set_ylabel(ylabel)

    fig.suptitle("qat 비교")
    add_language_legend(fig, make_solved_count_labels(solved_problem_indices_by_language, unsolved_problem_indices_by_language))
    save_figure(fig, model_name, f"qat_comparison_{config.benchmark}")


if __name__ == "__main__":
    main()
