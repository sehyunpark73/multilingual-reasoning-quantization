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

RESULT_DIRS = {"native": "understanding_insert/mgsm", "en": "en_understanding_insert/mgsm"}


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    method: Literal["bf16", "gptq", "awq", "nf4", "mxfp4"] = "bf16"
    calibration_languages: tuple[Literal["en", "zh", "ja", "sw", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_chunks: int = 128
    understanding_language: Literal["native", "en"] = "native"
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def _tokenize_with_understandings(
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    prompts: list[str],
    bf16_reasonings: list[str],
) -> tuple[list[int], list[Tensor]]:
    understandings = [extract_understanding(reasoning) for reasoning in bf16_reasonings]
    inserted_problem_indices = [problem_index for problem_index, understanding in enumerate(understandings) if understanding is not None]
    input_ids_list = []
    for problem_index in inserted_problem_indices:
        prompt_ids = tokenize_prompt(tokenizer, prompts[problem_index])
        understanding_ids = tokenize_reasoning(tokenizer, model_name, understandings[problem_index])
        input_ids_list.append(torch.cat([prompt_ids, understanding_ids]))
    return inserted_problem_indices, input_ids_list


def main() -> None:
    config = parse(Config)
    result_dir = RESULT_DIRS[config.understanding_language]
    hub_path = make_hub_path(config.method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_chunks)

    results = load_results(result_dir, hub_path, config.max_generation_len, config.seeds)
    missing_seeds = [seed for seed, result in results.items() if not result]
    if not missing_seeds:
        print(f"{hub_path} already evaluated")
        return

    problems = load_mgsm()
    bf16_results = load_results("loss/mgsm", make_hub_path("bf16", config.model_name), config.max_generation_len, tuple(missing_seeds))
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    seed_languages = [(seed, language) for seed in missing_seeds for language in LANGUAGES]
    inserted_problem_indices = {}
    input_ids_lists = []
    for seed, language in seed_languages:
        bf16_language = language if config.understanding_language == "native" else "en"
        inserted_problem_indices[seed, language], input_ids_list = _tokenize_with_understandings(tokenizer, config.model_name, problems[language]["prompts"], bf16_results[seed][bf16_language]["reasonings"])
        input_ids_lists.append(input_ids_list)

    model_path = config.model_name if config.method == "bf16" else download_model(hub_path)
    input_ids_list_seeds = [seed for seed, _ in seed_languages]
    generations = continue_generation(model_path, config.model_name, input_ids_lists, input_ids_list_seeds, config.max_generation_len)

    for (seed, language), generation in zip(seed_languages, generations):
        answers = [problems[language]["answers"][problem_index] for problem_index in inserted_problem_indices[seed, language]]
        results[seed][language] = {**score_generation(generation, answers, "mgsm", language), "inserted_problem_indices": inserted_problem_indices[seed, language]}
        print(f"{hub_path} seed {seed} {language} accuracy {results[seed][language]['accuracy']:.2%} reasoning len {mean(generation['reasoning_lens']):.0f} inserted {len(answers)} of {len(problems[language]['prompts'])}")

    for seed in missing_seeds:
        save_result(result_dir, hub_path, config.max_generation_len, seed, results[seed])


if __name__ == "__main__":
    main()
