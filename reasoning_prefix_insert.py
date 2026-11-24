from dataclasses import dataclass
from statistics import mean
from typing import Literal

from simple_parsing import parse
import torch
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from utils.dataset import LANGUAGES, load_mgsm
from utils.generate import continue_generation
from utils.hub import download_model, load_results, make_hub_path, save_result
from utils.score import score_generation
from utils.tokenize import tokenize_prompt, tokenize_reasoning
from utils.understanding import extract_understanding

RESULT_DIR = "reasoning_prefix_insert/mgsm"
LENGTH_RATIOS = [0.5, 0.75, 1, 1.5, 2]


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    method: Literal["gptq", "awq", "nf4", "mxfp4"] = "mxfp4"
    calibration_languages: tuple[Literal["en", "zh", "ja", "sw", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_chunks: int = 128
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def _tokenize_with_reasoning_prefixes(
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    prompts: list[str],
    bf16_reasonings: list[str],
    length_ratio: float,
) -> tuple[list[int], list[Tensor]]:
    inserted_problem_indices = []
    input_ids_list = []
    for problem_index, reasoning in enumerate(bf16_reasonings):
        understanding = extract_understanding(reasoning)
        if understanding is None:
            continue
        reasoning_ids = tokenize_reasoning(tokenizer, model_name, reasoning)
        understanding_len = len(tokenize_reasoning(tokenizer, model_name, understanding))
        if len(reasoning_ids) < max(LENGTH_RATIOS) * understanding_len:
            continue
        prompt_ids = tokenize_prompt(tokenizer, prompts[problem_index])
        inserted_problem_indices.append(problem_index)
        input_ids_list.append(torch.cat([prompt_ids, reasoning_ids[: round(length_ratio * understanding_len)]]))
    return inserted_problem_indices, input_ids_list


def main() -> None:
    config = parse(Config)
    hub_path = make_hub_path(config.method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_chunks)
    result_dirs = {length_ratio: f"{RESULT_DIR}/length_ratio={length_ratio}" for length_ratio in LENGTH_RATIOS}

    results = {length_ratio: load_results(result_dirs[length_ratio], hub_path, config.max_generation_len, config.seeds) for length_ratio in LENGTH_RATIOS}
    missing_seed_ratios = [(seed, length_ratio) for seed in config.seeds for length_ratio in LENGTH_RATIOS if not results[length_ratio][seed]]
    if not missing_seed_ratios:
        print(f"{hub_path} already evaluated")
        return

    problems = load_mgsm()
    missing_seeds = tuple(sorted({seed for seed, _ in missing_seed_ratios}))
    bf16_results = load_results("loss/mgsm", make_hub_path("bf16", config.model_name), config.max_generation_len, missing_seeds)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    seed_ratio_languages = [(seed, length_ratio, language) for seed, length_ratio in missing_seed_ratios for language in LANGUAGES]
    inserted_problem_indices = {}
    input_ids_lists = []
    for seed, length_ratio, language in seed_ratio_languages:
        inserted_problem_indices[seed, length_ratio, language], input_ids_list = _tokenize_with_reasoning_prefixes(tokenizer, config.model_name, problems[language]["prompts"], bf16_results[seed][language]["reasonings"], length_ratio)
        input_ids_lists.append(input_ids_list)

    input_ids_list_seeds = [seed for seed, _, _ in seed_ratio_languages]
    generations = continue_generation(download_model(hub_path), config.model_name, input_ids_lists, input_ids_list_seeds, config.max_generation_len)

    for (seed, length_ratio, language), generation in zip(seed_ratio_languages, generations):
        answers = [problems[language]["answers"][problem_index] for problem_index in inserted_problem_indices[seed, length_ratio, language]]
        results[length_ratio][seed][language] = {**score_generation(generation, answers, "mgsm", language), "inserted_problem_indices": inserted_problem_indices[seed, length_ratio, language]}
        print(f"{hub_path} seed {seed} length ratio {length_ratio} {language} accuracy {results[length_ratio][seed][language]['accuracy']:.2%} reasoning len {mean(generation['reasoning_lens']):.0f} inserted {len(answers)} of {len(problems[language]['prompts'])}")

    for seed, length_ratio in missing_seed_ratios:
        save_result(result_dirs[length_ratio], hub_path, config.max_generation_len, seed, results[length_ratio][seed])


if __name__ == "__main__":
    main()
