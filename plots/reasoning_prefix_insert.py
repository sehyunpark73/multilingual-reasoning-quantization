from dataclasses import dataclass
from statistics import mean

import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.dataset import LANGUAGES
from utils.hub import load_bf16_results, load_results
from utils.path import extract_model_name, make_result_dir
from utils.plot import INSERT_LOSS_YLIM, LANGUAGE_NAMES, add_language_legend, extract_method, make_insert_handles, plot_insert_bars, save_figure
from utils.score import compute_insert_losses
from utils.understanding import LENGTH_RATIOS

LENGTH_RATIO_NAMES = ["understanding 0.5배", "understanding 0.75배", "understanding 1배", "understanding 1.5배", "understanding 2배"]
FIGURE_SIZE = (12, 13)


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
    result_dir = make_result_dir("reasoning_prefix_insert", "mgsm")

    fig, axes = plt.subplots(len(config.model_hub_paths), 1, figsize=FIGURE_SIZE, sharey=True, squeeze=False, layout="constrained")
    for ax, model_hub_path in zip(axes[:, 0], config.model_hub_paths):
        no_insert_results = load_results(make_result_dir("loss", "mgsm"), model_hub_path, config.max_generation_len, config.seeds)
        insert_results_by_ratio = {length_ratio: load_results(f"{result_dir}/length_ratio={length_ratio}", model_hub_path, config.max_generation_len, config.seeds) for length_ratio in LENGTH_RATIOS}
        losses = {language: [] for language in LANGUAGES}
        num_problems = {}
        for language in LANGUAGES:
            problem_indices_by_seed = {seed: set(result[language]["inserted_problem_indices"]) for seed, result in insert_results_by_ratio[LENGTH_RATIOS[0]].items()}
            num_problems[language] = round(mean(len(problem_indices) for problem_indices in problem_indices_by_seed.values()))
            for length_ratio in LENGTH_RATIOS:
                losses[language].append(compute_insert_losses(bf16_results, no_insert_results, insert_results_by_ratio[length_ratio], language, problem_indices_by_seed))
        plot_insert_bars(ax, LENGTH_RATIO_NAMES, losses, [])
        ax.set_title(extract_method(model_hub_path))
        ax.set_ylabel("손실")
    axes[0, 0].set_ylim(INSERT_LOSS_YLIM)

    fig.suptitle("reasoning 앞부분 길이별 삽입")
    labels = {language: f"{LANGUAGE_NAMES[language]} {num_problems[language]}문제" for language in LANGUAGES}
    add_language_legend(fig, labels, make_insert_handles())
    save_figure(fig, model_name, "reasoning_prefix_insert")


if __name__ == "__main__":
    main()
