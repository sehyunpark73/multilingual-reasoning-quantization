from dataclasses import dataclass
import re
from statistics import mean

import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.dataset import LANGUAGES, load_mgsm
from utils.hub import load_bf16_results, load_results
from utils.path import extract_method, extract_model_name, make_result_dir
from utils.plot import LANGUAGE_NAMES, add_language_legend, compute_figure_size, make_insert_handles, plot_insert_bars, plot_language_bars, save_figure
from utils.score import compute_accuracy_loss_pairs

EN_QUESTION_WINDOW_RATIO = 4
MIN_EN_QUESTION_WORD_RATIO = 0.7
NO_INSTRUCTION_NAME = "instruction 없음\nbf16"


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


def _extract_words(
    text: str,
) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _compute_translation_ratio(
    reasonings: list[str],
    en_questions: list[str],
) -> float:
    num_translations = 0
    for reasoning, en_question in zip(reasonings, en_questions):
        en_words = _extract_words(en_question)
        window_words = _extract_words(reasoning[: EN_QUESTION_WINDOW_RATIO * len(en_question)])
        if len(en_words & window_words) / len(en_words) >= MIN_EN_QUESTION_WORD_RATIO:
            num_translations += 1
    return num_translations / len(reasonings)


def _compute_translation_ratios(
    all_results: list[dict[int, dict]],
    en_questions: list[str],
) -> dict[str, list[float]]:
    translation_ratios = {}
    for language in LANGUAGES:
        translation_ratios[language] = [mean(_compute_translation_ratio(result[language]["reasonings"], en_questions) for result in results.values()) for results in all_results]
    return translation_ratios


def main() -> None:
    config = parse(Config)
    model_name = extract_model_name(config.model_hub_paths[0])
    methods = [extract_method(model_hub_path) for model_hub_path in config.model_hub_paths]
    bf16_results = load_bf16_results("mgsm", model_name, config.max_generation_len, config.seeds)
    all_no_instruction_results = [load_results(make_result_dir("loss", "mgsm"), model_hub_path, config.max_generation_len, config.seeds) for model_hub_path in config.model_hub_paths]
    all_instruction_results = [load_results(make_result_dir("translate_first", "mgsm"), model_hub_path, config.max_generation_len, config.seeds) for model_hub_path in config.model_hub_paths]
    en_questions = load_mgsm()["en"]["questions"]

    fig, axes = plt.subplots(2, 1, figsize=compute_figure_size(2), layout="constrained")
    plot_insert_bars(axes[0], methods, compute_accuracy_loss_pairs(bf16_results, all_no_instruction_results, all_instruction_results), [])
    axes[0].set_ylabel("손실")
    translation_ratios = _compute_translation_ratios([bf16_results] + all_instruction_results, en_questions)
    plot_language_bars(axes[1], [NO_INSTRUCTION_NAME] + methods, translation_ratios, "%")
    axes[1].set_ylabel("번역 씀")

    fig.suptitle("번역 먼저 쓰기")
    add_language_legend(fig, LANGUAGE_NAMES, make_insert_handles("instruction 없음", "instruction 있음"))
    save_figure(fig, model_name, "translate_first")


if __name__ == "__main__":
    main()
