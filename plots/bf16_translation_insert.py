from dataclasses import dataclass

import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.hub import load_bf16_results, load_results
from utils.path import extract_method, extract_model_name, make_result_dir
from utils.plot import LANGUAGE_NAMES, add_language_legend, compute_figure_size, make_insert_handles, plot_insert_bars, save_figure
from utils.score import compute_accuracy_loss_pairs


@dataclass
class Config:
    model_hub_paths: tuple[str, ...] = (
        "bf16/google/gemma-4-E2B-it",
        "mxfp4/google/gemma-4-E2B-it",
    )
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def main() -> None:
    config = parse(Config)
    model_name = extract_model_name(config.model_hub_paths[0])
    methods = [extract_method(model_hub_path) for model_hub_path in config.model_hub_paths]
    bf16_results = load_bf16_results("mgsm", model_name, config.max_generation_len, config.seeds)
    all_no_insert_results = [load_results(make_result_dir("loss", "mgsm"), model_hub_path, config.max_generation_len, config.seeds) for model_hub_path in config.model_hub_paths]
    all_insert_results = [load_results(make_result_dir("bf16_translation_insert", "mgsm"), model_hub_path, config.max_generation_len, config.seeds) for model_hub_path in config.model_hub_paths]

    fig, ax = plt.subplots(figsize=compute_figure_size(1), layout="constrained")
    plot_insert_bars(ax, methods, compute_accuracy_loss_pairs(bf16_results, all_no_insert_results, all_insert_results), [])
    ax.set_ylabel("손실")

    fig.suptitle("bf16 번역 삽입")
    add_language_legend(fig, LANGUAGE_NAMES, make_insert_handles("삽입 없음", "삽입"))
    save_figure(fig, model_name, "bf16_translation_insert")


if __name__ == "__main__":
    main()
