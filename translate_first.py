from dataclasses import dataclass
from statistics import mean

from simple_parsing import parse

from utils.dataset import LANGUAGES, load_mgsm
from utils.generate import generate
from utils.hub import download_model, load_results, save_result
from utils.path import extract_model_name, make_result_dir
from utils.score import score_generation

TRANSLATE_FIRST_INSTRUCTION = "Start your thinking by writing the problem above in English, then solve it."


@dataclass
class Config:
    model_hub_path: str = "bf16/google/gemma-4-E2B-it"
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def main() -> None:
    config = parse(Config)
    model_name = extract_model_name(config.model_hub_path)
    result_dir = make_result_dir("translate_first", "mgsm")

    results = load_results(result_dir, config.model_hub_path, config.max_generation_len, config.seeds)
    missing_seeds = [seed for seed, result in results.items() if not result]
    if not missing_seeds:
        print(f"{config.model_hub_path} already evaluated")
        return

    problems = load_mgsm()

    missing_seed_languages = [(seed, language) for seed in missing_seeds for language in LANGUAGES]
    prompt_lists = []
    for _, language in missing_seed_languages:
        prompt_lists.append([f"{prompt}\n\n{TRANSLATE_FIRST_INSTRUCTION}" for prompt in problems[language]["prompts"]])
    prompt_list_seeds = [seed for seed, _ in missing_seed_languages]
    generations = generate(download_model(config.model_hub_path), model_name, prompt_lists, prompt_list_seeds, config.max_generation_len)

    for (seed, language), generation in zip(missing_seed_languages, generations):
        results[seed][language] = score_generation(generation, problems[language]["answers"], "mgsm", language)
        print(f"{config.model_hub_path} seed {seed} {language} accuracy {results[seed][language]['accuracy']:.2%} reasoning len {mean(generation['reasoning_lens']):.0f} truncated {mean(generation['is_truncateds']):.2%}")

    for seed in missing_seeds:
        save_result(result_dir, config.model_hub_path, config.max_generation_len, seed, results[seed])


if __name__ == "__main__":
    main()
