from dataclasses import dataclass

import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.dataset import LANGUAGES
from utils.hub import load_bf16_results, load_results
from utils.path import extract_method, extract_model_name, make_result_dir
from utils.plot import add_language_legend, compute_figure_size, make_insert_handles, make_problem_count_labels, plot_insert_bars, save_figure
from utils.score import compute_insert_losses, find_common_inserted_problem_indices

FIGURE_WIDTH = 15


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
        insert_results = load_results(make_result_dir("en_understanding_insert", "mgsm"), model_hub_path, config.max_generation_len, config.seeds)
        for language in LANGUAGES:
            losses[language].append(compute_insert_losses(bf16_results, no_insert_results, insert_results, language, problem_indices_by_language[language]))

    fig, ax = plt.subplots(figsize=compute_figure_size(1, FIGURE_WIDTH), layout="constrained")
    plot_insert_bars(ax, methods, losses, [])
    ax.set_ylabel("손실")

    fig.suptitle("영어 understanding 삽입")
    add_language_legend(fig, make_problem_count_labels(problem_indices_by_language), make_insert_handles("삽입 없음", "삽입"))
    save_figure(fig, model_name, "en_understanding_insert")


if __name__ == "__main__":
    main()
