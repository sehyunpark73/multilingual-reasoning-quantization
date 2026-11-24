from dataclasses import dataclass
from statistics import mean
from typing import Literal

from simple_parsing import parse
import torch
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from utils.dataset import LANGUAGES, load_mgsm
from utils.generate import continue_generation
from utils.hub import download_model, load_result, load_results, save_result
from utils.model import MODEL_CONFIGS
from utils.path import extract_model_name
from utils.score import score_generation
from utils.tokenize import tokenize_prompt, tokenize_reasoning

RESULT_DIRS = {"mgsm": "en_question_insert/mgsm", "bf16_translation": "bf16_translation_insert/mgsm"}


@dataclass
class Config:
    model_hub_path: str = "bf16/google/gemma-4-E2B-it"
    en_question: Literal["mgsm", "bf16_translation"] = "mgsm"
    translation_hub_path: str = "mgsm_translation/bf16/google/gemma-4-E2B-it/max_generation_len=512,seed=0.json"
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def _load_en_questions(
    problems: dict[str, dict[str, list]],
    en_question: str,
    translation_hub_path: str,
) -> dict[str, list[str]]:
    if en_question == "mgsm":
        return {language: problems["en"]["questions"] for language in LANGUAGES}
    translation_result = load_result(translation_hub_path)
    en_questions = {"en": problems["en"]["questions"]}
    for language in translation_result:
        en_questions[language] = translation_result[language]["en_questions"]
    return en_questions


def _tokenize_with_en_questions(
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    prompts: list[str],
    en_questions: list[str],
) -> list[Tensor]:
    input_ids_list = []
    for prompt, en_question in zip(prompts, en_questions):
        prompt_ids = tokenize_prompt(tokenizer, prompt)
        en_question_ids = tokenize_reasoning(tokenizer, model_name, f"{MODEL_CONFIGS[model_name]['reasoning_header']}{en_question}\n\n")
        input_ids_list.append(torch.cat([prompt_ids, en_question_ids]))
    return input_ids_list


def main() -> None:
    config = parse(Config)
    model_name = extract_model_name(config.model_hub_path)
    result_dir = RESULT_DIRS[config.en_question]

    results = load_results(result_dir, config.model_hub_path, config.max_generation_len, config.seeds)
    missing_seeds = [seed for seed, result in results.items() if not result]
    if not missing_seeds:
        print(f"{config.model_hub_path} already evaluated")
        return

    problems = load_mgsm()
    en_questions = _load_en_questions(problems, config.en_question, config.translation_hub_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids_lists_by_language = {language: _tokenize_with_en_questions(tokenizer, model_name, problems[language]["prompts"], en_questions[language]) for language in LANGUAGES}

    missing_seed_languages = [(seed, language) for seed in missing_seeds for language in LANGUAGES]
    input_ids_lists = [input_ids_lists_by_language[language] for _, language in missing_seed_languages]
    input_ids_list_seeds = [seed for seed, _ in missing_seed_languages]
    generations = continue_generation(download_model(config.model_hub_path), model_name, input_ids_lists, input_ids_list_seeds, config.max_generation_len)

    for (seed, language), generation in zip(missing_seed_languages, generations):
        results[seed][language] = score_generation(generation, problems[language]["answers"], "mgsm", language)
        print(f"{config.model_hub_path} seed {seed} {language} accuracy {results[seed][language]['accuracy']:.2%} reasoning len {mean(generation['reasoning_lens']):.0f} truncated {mean(generation['is_truncateds']):.2%}")

    for seed in missing_seeds:
        save_result(result_dir, config.model_hub_path, config.max_generation_len, seed, results[seed])


if __name__ == "__main__":
    main()
