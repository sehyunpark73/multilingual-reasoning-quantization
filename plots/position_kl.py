from dataclasses import dataclass

import matplotlib.pyplot as plt
from simple_parsing import parse
import torch
from torch import Tensor

from utils.dataset import LANGUAGES
from utils.hub import load_bf16_results, load_tensors
from utils.path import extract_model_name, make_result_dir
from utils.plot import LANGUAGE_COLORS, LANGUAGE_NAMES, NUM_POSITION_BINS, POSITION_LABELS, add_language_legend, extract_method, save_figure
from utils.score import find_solved_problem_indices

FIGURE_SIZE = (12, 3.5)


@dataclass
class Config:
    model_hub_paths: tuple[str, ...] = (
        "gptq/google/gemma-4-E2B-it/calibration_languages=en,num_bits=4,group_size=128,num_chunks=128",
        "awq/google/gemma-4-E2B-it/calibration_languages=en,num_bits=4,group_size=128,num_chunks=128",
        "nf4/google/gemma-4-E2B-it",
        "mxfp4/google/gemma-4-E2B-it",
    )
    max_generation_len: int = 16384
    reasoning_seed: int = 0
    bf16_seeds: tuple[int, ...] = (0, 1, 2)


def _compute_position_kls(
    kls: Tensor,
    token_problem_indices: Tensor,
    solved_problem_indices: list[int],
) -> list[float]:
    problem_indices, num_tokens = token_problem_indices.unique_consecutive(return_counts=True)
    problem_position_kls = []
    for problem_index, problem_kls in zip(problem_indices.tolist(), kls.split(num_tokens.tolist())):
        if problem_index in solved_problem_indices:
            bin_kls = [position_bin_kls.mean() for position_bin_kls in problem_kls.tensor_split(NUM_POSITION_BINS)]
            problem_position_kls.append(torch.stack(bin_kls))
    return torch.stack(problem_position_kls).mean(dim=0).tolist()


def main() -> None:
    config = parse(Config)
    model_name = extract_model_name(config.model_hub_paths[0])
    bf16_results = load_bf16_results("mgsm", model_name, config.max_generation_len, config.bf16_seeds)
    solved_problem_indices = {language: find_solved_problem_indices(bf16_results, language) for language in LANGUAGES}

    fig, axes = plt.subplots(1, len(config.model_hub_paths), figsize=FIGURE_SIZE, sharey=True, squeeze=False, layout="constrained")
    for ax, model_hub_path in zip(axes[0], config.model_hub_paths):
        tensors = load_tensors(make_result_dir("position_kl", "mgsm"), model_hub_path, config.max_generation_len, config.reasoning_seed)
        for language in LANGUAGES:
            position_kls = _compute_position_kls(tensors[f"{language}/kls"], tensors[f"{language}/token_problem_indices"], solved_problem_indices[language])
            ax.plot(POSITION_LABELS, position_kls, color=LANGUAGE_COLORS[language], marker="o", linewidth=2)
        ax.set_title(extract_method(model_hub_path))
        ax.tick_params(axis="x", labelrotation=45)
    axes[0, 0].set_ylabel("kl")

    fig.suptitle("reasoning 위치별 kl")
    labels = {language: f"{LANGUAGE_NAMES[language]} {len(solved_problem_indices[language])}문제" for language in LANGUAGES}
    add_language_legend(fig, labels)
    save_figure(fig, model_name, "position_kl")


if __name__ == "__main__":
    main()
