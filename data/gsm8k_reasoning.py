from dataclasses import dataclass

from simple_parsing import parse

from utils.dataset import LANGUAGES, load_translated_gsm8k
from utils.generate import generate
from utils.hub import has_result, save_result
from utils.path import make_model_hub_path
from utils.score import score_generation

RESULT_DIR = "gsm8k_reasoning"


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    max_generation_len: int = 16384
    seed: int = 0


def main() -> None:
    config = parse(Config)
    model_hub_path = make_model_hub_path("bf16", config.model_name)

    if has_result(RESULT_DIR, model_hub_path, config.max_generation_len, config.seed):
        print(f"{model_hub_path} already generated")
        return

    problems = load_translated_gsm8k()
    prompt_lists = [problems[language]["prompts"] for language in LANGUAGES]
    prompt_list_seeds = [config.seed] * len(LANGUAGES)
    generations = generate(config.model_name, config.model_name, prompt_lists, prompt_list_seeds, config.max_generation_len, is_reasoning_ids_returned=True)

    full_result = {}
    for language, generation in zip(LANGUAGES, generations):
        full_result[language] = score_generation(generation, problems[language]["answers"], "mgsm", language)
        print(f"{model_hub_path} {language} accuracy {full_result[language]['accuracy']:.2%}")

    en_correct_problem_indices = [problem_index for problem_index, is_correct in enumerate(full_result["en"]["is_corrects"]) if is_correct]
    result = {}
    for language in LANGUAGES:
        result[language] = {
            "prompts": [problems[language]["prompts"][problem_index] for problem_index in en_correct_problem_indices],
            "reasonings": [full_result[language]["reasonings"][problem_index] for problem_index in en_correct_problem_indices],
            "reasoning_ids": [full_result[language]["reasoning_ids"][problem_index] for problem_index in en_correct_problem_indices],
            "is_corrects": [full_result[language]["is_corrects"][problem_index] for problem_index in en_correct_problem_indices],
        }

    print(f"{model_hub_path} en correct {len(en_correct_problem_indices)} of {len(full_result['en']['is_corrects'])}")
    save_result(RESULT_DIR, model_hub_path, config.max_generation_len, config.seed, result)


if __name__ == "__main__":
    main()
