from dataclasses import dataclass

from simple_parsing import parse

from utils.dataset import LANGUAGES, load_gsm8k
from utils.generate import generate
from utils.hub import load_results, make_hub_path, save_result
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
    if load_results(RESULT_DIR, hub_path, config.max_generation_len, (config.seed,))[config.seed]:
        print(f"{hub_path} already generated")
        return

    problems = load_gsm8k()
    generations = generate(config.model_name, config.model_name, [problems[language]["prompts"] for language in LANGUAGES], [config.seed] * len(LANGUAGES), config.max_generation_len)
    corrects = {language: score_generation(generation, problems[language]["answers"], "mgsm", language)["corrects"] for language, generation in zip(LANGUAGES, generations)}

    en_solved_problem_indices = [problem_index for problem_index, correct in enumerate(corrects["en"]) if correct]
    result = {}
    for language, generation in zip(LANGUAGES, generations):
        result[language] = {
            "prompts": [problems[language]["prompts"][problem_index] for problem_index in en_solved_problem_indices],
            "reasonings": [generation["reasonings"][problem_index] for problem_index in en_solved_problem_indices],
            "corrects": [corrects[language][problem_index] for problem_index in en_solved_problem_indices],
        }
        print(f"{hub_path} {language} accuracy {sum(corrects[language]) / len(corrects[language]):.2%}")
    print(f"{hub_path} en solved {len(en_solved_problem_indices)} of {len(corrects['en'])}")
    save_result(RESULT_DIR, hub_path, config.max_generation_len, config.seed, result)


if __name__ == "__main__":
    main()
