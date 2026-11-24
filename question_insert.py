from dataclasses import dataclass
from statistics import mean
from typing import Literal, TypeAlias

from simple_parsing import parse
from transformers import AutoTokenizer

from utils.dataset import LANGUAGES, NON_EN_LANGUAGES, load_mgsm
from utils.generate import continue_generation
from utils.hub import download_model, load_result, load_results, save_result
from utils.path import extract_model_name, make_result_dir
from utils.score import score_generation
from utils.tokenize import tokenize_with_en_questions

_EnQuestion: TypeAlias = Literal["mgsm", "bf16_translation"]


@dataclass
class Config:
    model_hub_path: str = "bf16/google/gemma-4-E2B-it"
    en_question: _EnQuestion = "mgsm"
    translation_hub_path: str = "mgsm_translation/bf16/google/gemma-4-E2B-it/max_generation_len=512,seed=0.json"
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def _load_en_questions(
    problems: dict[str, dict[str, list]],
    en_question: _EnQuestion,
    translation_hub_path: str,
) -> dict[str, list[str]]:
    if en_question == "mgsm":
        return {language: problems["en"]["questions"] for language in LANGUAGES}
    if en_question == "bf16_translation":
        translation_result = load_result(translation_hub_path)
        en_questions = {"en": problems["en"]["questions"]}
        for language in NON_EN_LANGUAGES:
            en_questions[language] = translation_result[language]["en_questions"]
        return en_questions


def main() -> None:
    config = parse(Config)
    model_name = extract_model_name(config.model_hub_path)
    if config.en_question == "mgsm":
        result_dir = make_result_dir("en_question_insert", "mgsm")
    if config.en_question == "bf16_translation":
        result_dir = make_result_dir("bf16_translation_insert", "mgsm")

    results = load_results(result_dir, config.model_hub_path, config.max_generation_len, config.seeds)
    missing_seeds = [seed for seed, result in results.items() if not result]
    if not missing_seeds:
        print(f"{config.model_hub_path} already evaluated")
        return

    problems = load_mgsm()
    en_questions = _load_en_questions(problems, config.en_question, config.translation_hub_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    missing_seed_languages = [(seed, language) for seed in missing_seeds for language in LANGUAGES]
    input_ids_lists = []
    for _, language in missing_seed_languages:
        input_ids_lists.append(tokenize_with_en_questions(tokenizer, model_name, problems[language]["prompts"], en_questions[language]))
    input_ids_list_seeds = [seed for seed, _ in missing_seed_languages]
    generations = continue_generation(download_model(config.model_hub_path), model_name, input_ids_lists, input_ids_list_seeds, config.max_generation_len)

    for (seed, language), generation in zip(missing_seed_languages, generations):
        results[seed][language] = score_generation(generation, problems[language]["answers"], "mgsm", language)
        print(f"{config.model_hub_path} seed {seed} {language} accuracy {results[seed][language]['accuracy']:.2%} reasoning len {mean(generation['reasoning_lens']):.0f} truncated {mean(generation['is_truncateds']):.2%}")

    for seed in missing_seeds:
        save_result(result_dir, config.model_hub_path, config.max_generation_len, seed, results[seed])


if __name__ == "__main__":
    main()
