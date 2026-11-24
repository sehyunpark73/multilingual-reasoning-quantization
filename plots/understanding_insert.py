from dataclasses import dataclass
from statistics import mean

import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.dataset import LANGUAGES
from utils.hub import load_bf16_results, load_results
from utils.path import extract_model_name, make_result_dir
from utils.plot import INSERT_LOSS_YLIM, LANGUAGE_NAMES, add_language_legend, extract_method, make_insert_handles, plot_insert_bars, save_figure
from utils.score import compute_insert_losses, find_common_inserted_problem_indices

RECOVERY_METHODS = ["mxfp4"]
FIGURE_SIZE = (15, 5)


@dataclass
class Config:
    model_hub_paths: tuple[str, ...] = (
        "bf16/google/gemma-4-E2B-it",
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
    methods = [extract_method(model_hub_path) for model_hub_path in config.model_hub_paths]
    bf16_results = load_bf16_results("mgsm", model_name, config.max_generation_len, config.seeds)
    native_results = load_results(make_result_dir("understanding_insert", "mgsm"), config.model_hub_paths[0], config.max_generation_len, config.seeds)
    en_results = load_results(make_result_dir("en_understanding_insert", "mgsm"), config.model_hub_paths[0], config.max_generation_len, config.seeds)
    problem_indices_by_language = {language: find_common_inserted_problem_indices(native_results, en_results, language) for language in LANGUAGES}

    losses = {language: [] for language in LANGUAGES}
    for model_hub_path in config.model_hub_paths:
        no_insert_results = load_results(make_result_dir("loss", "mgsm"), model_hub_path, config.max_generation_len, config.seeds)
        insert_results = load_results(make_result_dir("understanding_insert", "mgsm"), model_hub_path, config.max_generation_len, config.seeds)
        for language in LANGUAGES:
            losses[language].append(compute_insert_losses(bf16_results, no_insert_results, insert_results, language, problem_indices_by_language[language]))

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, layout="constrained")
    plot_insert_bars(ax, methods, losses, RECOVERY_METHODS)
    ax.set_ylabel("손실")
    ax.set_ylim(INSERT_LOSS_YLIM)

    fig.suptitle("understanding 삽입")
    labels = {}
    for language, problem_indices_by_seed in problem_indices_by_language.items():
        num_problems = round(mean(len(problem_indices) for problem_indices in problem_indices_by_seed.values()))
        labels[language] = f"{LANGUAGE_NAMES[language]} {num_problems}문제"
    add_language_legend(fig, labels, make_insert_handles())
    save_figure(fig, model_name, "understanding_insert")


if __name__ == "__main__":
    main()
