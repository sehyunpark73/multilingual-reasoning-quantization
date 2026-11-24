from dataclasses import dataclass

from simple_parsing import parse

from utils.dataset import LANGUAGES, load_translated_gsm8k
from utils.generate import generate
from utils.hub import make_hub_path, result_exists, save_result
from utils.score import score_generation

RESULT_DIR = "gsm8k_reasoning"


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    max_generation_len: int = 16384
    seed: int = 0


def main() -> None:
    config = parse(Config)
    hub_path = make_hub_path("bf16", config.model_name)
    if result_exists(RESULT_DIR, hub_path, config.max_generation_len, config.seed):
        print(f"{hub_path} already generated")
        return

    problems = load_translated_gsm8k()
    prompt_lists = [problems[language]["prompts"] for language in LANGUAGES]
    prompt_list_seeds = [config.seed] * len(LANGUAGES)
    generations = generate(config.model_name, config.model_name, prompt_lists, prompt_list_seeds, config.max_generation_len)
    full_result = {}
    for language, generation in zip(LANGUAGES, generations):
        full_result[language] = score_generation(generation, problems[language]["answers"], "mgsm", language)
        print(f"{hub_path} {language} accuracy {full_result[language]['accuracy']:.2%}")

    en_correct_problem_indices = [problem_index for problem_index, correct in enumerate(full_result["en"]["corrects"]) if correct]
    result = {}
    for language in LANGUAGES:
        result[language] = {
            "prompts": [problems[language]["prompts"][problem_index] for problem_index in en_correct_problem_indices],
            "reasonings": [full_result[language]["reasonings"][problem_index] for problem_index in en_correct_problem_indices],
            "corrects": [full_result[language]["corrects"][problem_index] for problem_index in en_correct_problem_indices],
        }
    print(f"{hub_path} en correct {len(en_correct_problem_indices)} of {len(full_result['en']['corrects'])}")
    save_result(RESULT_DIR, hub_path, config.max_generation_len, config.seed, result)


if __name__ == "__main__":
    main()
