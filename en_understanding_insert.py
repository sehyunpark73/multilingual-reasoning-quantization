from dataclasses import dataclass
from statistics import mean

from simple_parsing import parse
from transformers import AutoTokenizer

from utils.dataset import LANGUAGES, load_mgsm
from utils.generate import continue_generation
from utils.hub import download_model, load_bf16_results, load_results, save_result
from utils.path import extract_model_name, make_result_dir
from utils.score import score_generation
from utils.tokenize import tokenize_with_understandings


@dataclass
class Config:
    model_hub_path: str = "bf16/google/gemma-4-E2B-it"
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)


def main() -> None:
    config = parse(Config)
    model_name = extract_model_name(config.model_hub_path)
    result_dir = make_result_dir("en_understanding_insert", "mgsm")

    results = load_results(result_dir, config.model_hub_path, config.max_generation_len, config.seeds)
    missing_seeds = [seed for seed, result in results.items() if not result]
    if not missing_seeds:
        print(f"{config.model_hub_path} already evaluated")
        return

    problems = load_mgsm()
    bf16_results = load_bf16_results("mgsm", model_name, config.max_generation_len, missing_seeds)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    missing_seed_languages = [(seed, language) for seed in missing_seeds for language in LANGUAGES]
    all_inserted_problem_indices = {}
    input_ids_lists = []
    for seed, language in missing_seed_languages:
        inserted_problem_indices, input_ids_list = tokenize_with_understandings(tokenizer, model_name, problems[language]["prompts"], bf16_results[seed]["en"]["reasonings"])
        all_inserted_problem_indices[seed, language] = inserted_problem_indices
        input_ids_lists.append(input_ids_list)

    input_ids_list_seeds = [seed for seed, _ in missing_seed_languages]
    generations = continue_generation(download_model(config.model_hub_path), model_name, input_ids_lists, input_ids_list_seeds, config.max_generation_len)

    for (seed, language), generation in zip(missing_seed_languages, generations):
        answers = [problems[language]["answers"][problem_index] for problem_index in all_inserted_problem_indices[seed, language]]
        results[seed][language] = {"inserted_problem_indices": all_inserted_problem_indices[seed, language], **score_generation(generation, answers, "mgsm", language)}
        print(f"{config.model_hub_path} seed {seed} {language} accuracy {results[seed][language]['accuracy']:.2%} reasoning len {mean(generation['reasoning_lens']):.0f} inserted {len(answers)} of {len(problems[language]['prompts'])}")

    for seed in missing_seeds:
        save_result(result_dir, config.model_hub_path, config.max_generation_len, seed, results[seed])


if __name__ == "__main__":
    main()
