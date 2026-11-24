from dataclasses import dataclass

import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.dataset import LANGUAGES
from utils.hub import load_bf16_results, load_results
from utils.path import extract_model_name, make_result_dir
from utils.plot import LANGUAGE_NAMES, add_language_legend, extract_method, plot_loss_bars, save_figure
from utils.score import compute_mean_accuracy

CALIBRATION_NAMES = ["영어", "스와힐리어", "텔루구어", "다섯 언어 혼합"]
FIGURE_SIZE = (10, 6)


@dataclass
class Config:
    gptq_model_hub_paths: tuple[str, ...] = (
        "gptq/google/gemma-4-E2B-it/calibration_languages=en,num_bits=4,group_size=128,num_chunks=128",
        "gptq/google/gemma-4-E2B-it/calibration_languages=sw,num_bits=4,group_size=128,num_chunks=128",
        "gptq/google/gemma-4-E2B-it/calibration_languages=te,num_bits=4,group_size=128,num_chunks=128",
        "gptq/google/gemma-4-E2B-it/calibration_languages=en_zh_ja_sw_te,num_bits=4,group_size=128,num_chunks=128",
    )
    awq_model_hub_paths: tuple[str, ...] = (
        "awq/google/gemma-4-E2B-it/calibration_languages=en,num_bits=4,group_size=128,num_chunks=128",
        "awq/google/gemma-4-E2B-it/calibration_languages=sw,num_bits=4,group_size=128,num_chunks=128",
        "awq/google/gemma-4-E2B-it/calibration_languages=te,num_bits=4,group_size=128,num_chunks=128",
        "awq/google/gemma-4-E2B-it/calibration_languages=en_zh_ja_sw_te,num_bits=4,group_size=128,num_chunks=128",
    )
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def _compute_losses(
    bf16_results: dict[int, dict],
    model_hub_paths: tuple[str, ...],
    max_generation_len: int,
    seeds: tuple[int, ...],
) -> dict[str, list[float]]:
    losses = {language: [] for language in LANGUAGES}
    for model_hub_path in model_hub_paths:
        results = load_results(make_result_dir("loss", "mgsm"), model_hub_path, max_generation_len, seeds)
        for language in LANGUAGES:
            losses[language].append(compute_mean_accuracy(bf16_results, language) - compute_mean_accuracy(results, language))
    return losses


def main() -> None:
    config = parse(Config)
    model_name = extract_model_name(config.gptq_model_hub_paths[0])
    bf16_results = load_bf16_results("mgsm", model_name, config.max_generation_len, config.seeds)

    fig, axes = plt.subplots(2, 1, figsize=FIGURE_SIZE, sharey=True, layout="constrained")
    for ax, model_hub_paths in zip(axes, [config.gptq_model_hub_paths, config.awq_model_hub_paths]):
        losses = _compute_losses(bf16_results, model_hub_paths, config.max_generation_len, config.seeds)
        plot_loss_bars(ax, CALIBRATION_NAMES, losses)
        ax.set_title(extract_method(model_hub_paths[0]))
        ax.set_ylabel("손실")

    fig.suptitle("calibration 언어별 손실")
    add_language_legend(fig, LANGUAGE_NAMES)
    save_figure(fig, model_name, "calibration")


if __name__ == "__main__":
    main()
