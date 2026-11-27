from dataclasses import dataclass

import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.dataset import LANGUAGES
from utils.hub import load_bf16_results, load_results
from utils.path import extract_method, extract_model_name, make_reasoning_prefix_result_dir, make_result_dir
from utils.plot import add_language_legend, compute_figure_size, make_insert_handles, make_problem_count_labels, plot_insert_bars, save_figure
from utils.score import compute_insert_losses
from utils.understanding import LENGTH_RATIOS

LENGTH_RATIO_NAMES = ["understanding 0.5배", "understanding 0.75배", "understanding 1배", "understanding 1.5배", "understanding 2배"]


@dataclass
class Config:
    model_hub_paths: tuple[str, ...] = (
        "gptq/google/gemma-4-E2B-it/calibration_languages=en,num_bits=4,group_size=128,num_chunks=128",
        "awq/google/gemma-4-E2B-it/calibration_languages=en,num_bits=4,group_size=128,num_chunks=128",
        "nf4/google/gemma-4-E2B-it",
        "mxfp4/google/gemma-4-E2B-it",
    )
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def main() -> None:
    config = parse(Config)
    model_name = extract_model_name(config.model_hub_paths[0])
    bf16_results = load_bf16_results("mgsm", model_name, config.max_generation_len, config.seeds)
    first_ratio_results = load_results(make_reasoning_prefix_result_dir(LENGTH_RATIOS[0]), config.model_hub_paths[0], config.max_generation_len, config.seeds)
    problem_indices_by_language = {}
    for language in LANGUAGES:
        problem_indices_by_language[language] = {seed: set(result[language]["inserted_problem_indices"]) for seed, result in first_ratio_results.items()}

    fig, axes = plt.subplots(len(config.model_hub_paths), 1, figsize=compute_figure_size(len(config.model_hub_paths)), squeeze=False, layout="constrained")
    for ax, model_hub_path in zip(axes[:, 0], config.model_hub_paths):
        no_insert_results = load_results(make_result_dir("loss", "mgsm"), model_hub_path, config.max_generation_len, config.seeds)
        insert_results_by_ratio = {length_ratio: load_results(make_reasoning_prefix_result_dir(length_ratio), model_hub_path, config.max_generation_len, config.seeds) for length_ratio in LENGTH_RATIOS}
        losses = {language: [] for language in LANGUAGES}
        for language in LANGUAGES:
            for length_ratio in LENGTH_RATIOS:
                losses[language].append(compute_insert_losses(bf16_results, no_insert_results, insert_results_by_ratio[length_ratio], language, problem_indices_by_language[language]))
        plot_insert_bars(ax, LENGTH_RATIO_NAMES, losses, [])
        ax.set_title(extract_method(model_hub_path))
        ax.set_ylabel("손실")

    fig.suptitle("reasoning 앞부분 길이별 삽입")
    add_language_legend(fig, make_problem_count_labels(problem_indices_by_language), make_insert_handles("삽입 없음", "삽입"))
    save_figure(fig, model_name, "reasoning_prefix_insert")


if __name__ == "__main__":
    main()
