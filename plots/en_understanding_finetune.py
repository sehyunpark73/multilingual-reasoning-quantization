from dataclasses import dataclass

import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.hub import load_bf16_results, load_results
from utils.path import extract_model_name, make_result_dir
from utils.plot import LANGUAGE_NAMES, add_language_legend, compute_figure_size, plot_language_bars, save_figure
from utils.score import compute_accuracy_losses

FINETUNE_NAMES = ["ce\nbf16 영어 understanding", "kl\nbf16 영어 understanding", "kl\non policy understanding", "kl\non policy reasoning", "hidden\non policy understanding"]


@dataclass
class Config:
    model_hub_paths: tuple[str, ...] = (
        "en_understanding_finetune/google/gemma-4-E2B-it/loss=ce,reasoning_source=bf16_en,train_part=understanding",
        "en_understanding_finetune/google/gemma-4-E2B-it/loss=kl,reasoning_source=bf16_en,train_part=understanding",
        "en_understanding_finetune/google/gemma-4-E2B-it/loss=kl,reasoning_source=on_policy,train_part=understanding",
        "en_understanding_finetune/google/gemma-4-E2B-it/loss=kl,reasoning_source=on_policy,train_part=reasoning",
        "en_understanding_finetune/google/gemma-4-E2B-it/loss=hidden,reasoning_source=on_policy,train_part=understanding",
    )
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def main() -> None:
    config = parse(Config)
    model_name = extract_model_name(config.model_hub_paths[0])
    bf16_results = load_bf16_results("mgsm", model_name, config.max_generation_len, config.seeds)
    all_results = [load_results(make_result_dir("loss", "mgsm"), model_hub_path, config.max_generation_len, config.seeds) for model_hub_path in config.model_hub_paths]

    fig, ax = plt.subplots(figsize=compute_figure_size(1), layout="constrained")
    plot_language_bars(ax, FINETUNE_NAMES, compute_accuracy_losses(bf16_results, all_results), "%p")
    ax.set_ylabel("손실")

    fig.suptitle("영어 understanding finetune")
    add_language_legend(fig, LANGUAGE_NAMES)
    save_figure(fig, model_name, "en_understanding_finetune")


if __name__ == "__main__":
    main()
