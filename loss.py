from dataclasses import dataclass
from statistics import mean
from typing import Literal

from simple_parsing import parse

from utils.dataset import LANGUAGES, load_mgsm, load_mmlu_prox_lite
from utils.generate import generate
from utils.hub import download_model, load_results, save_result
from utils.path import extract_model_name, make_loss_result_dir
from utils.score import score_generation


@dataclass
class Config:
    model_hub_path: str = "bf16/google/gemma-4-E2B-it"
    benchmark: Literal["mgsm", "mmlu_prox_lite"] = "mgsm"
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def _find_missing_seed_languages(
    results: dict[int, dict],
) -> list[tuple[int, str]]:
    missing_seed_languages = []
    for seed, result in results.items():
        for language in LANGUAGES:
            if language not in result:
                missing_seed_languages.append((seed, language))
    return missing_seed_languages


def main() -> None:
    config = parse(Config)
    model_name = extract_model_name(config.model_hub_path)

    result_dir = make_loss_result_dir(config.benchmark)
    results = load_results(result_dir, config.model_hub_path, config.max_generation_len, config.seeds)
    missing_seed_languages = _find_missing_seed_languages(results)
    if not missing_seed_languages:
        print(f"{config.model_hub_path} already evaluated")
        return

    problems = load_mgsm() if config.benchmark == "mgsm" else load_mmlu_prox_lite()
    prompt_lists = [problems[language]["prompts"] for _, language in missing_seed_languages]
    prompt_list_seeds = [seed for seed, _ in missing_seed_languages]
    generations = generate(download_model(config.model_hub_path), model_name, prompt_lists, prompt_list_seeds, config.max_generation_len)

    for (seed, language), generation in zip(missing_seed_languages, generations):
        results[seed][language] = score_generation(generation, problems[language]["answers"], config.benchmark, language)
        print(f"{config.model_hub_path} seed {seed} {language} accuracy {results[seed][language]['accuracy']:.2%} reasoning len {mean(generation['reasoning_lens']):.0f} truncated {mean(generation['is_truncateds']):.2%}")

    missing_seeds = sorted({seed for seed, _ in missing_seed_languages})
    for seed in missing_seeds:
        save_result(result_dir, config.model_hub_path, config.max_generation_len, seed, results[seed])


if __name__ == "__main__":
    main()
