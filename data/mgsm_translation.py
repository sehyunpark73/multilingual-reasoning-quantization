from dataclasses import dataclass
from statistics import mean

from simple_parsing import parse

from utils.dataset import NON_EN_LANGUAGES, load_mgsm
from utils.generate import generate_without_thinking
from utils.hub import has_result, save_result
from utils.path import make_model_hub_path

RESULT_DIR = "mgsm_translation"
TRANSLATION_INSTRUCTION = "Translate the following question into English. Output only the translation."


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    max_generation_len: int = 512
    seed: int = 0


def main() -> None:
    config = parse(Config)
    model_hub_path = make_model_hub_path("bf16", config.model_name)

    if has_result(RESULT_DIR, model_hub_path, config.max_generation_len, config.seed):
        print(f"{model_hub_path} already generated")
        return

    problems = load_mgsm()
    prompt_lists = []
    for language in NON_EN_LANGUAGES:
        prompt_lists.append([f"{TRANSLATION_INSTRUCTION}\n\n{question}" for question in problems[language]["questions"]])
    prompt_list_seeds = [config.seed] * len(NON_EN_LANGUAGES)
    generations = generate_without_thinking(config.model_name, config.model_name, prompt_lists, prompt_list_seeds, config.max_generation_len)

    result = {}
    for language, generation in zip(NON_EN_LANGUAGES, generations):
        result[language] = {"en_questions": [response.strip() for response in generation["responses"]], "is_truncateds": generation["is_truncateds"]}
        print(f"{model_hub_path} {language} truncated {mean(result[language]['is_truncateds']):.2%} example {result[language]['en_questions'][0]!r}")

    save_result(RESULT_DIR, model_hub_path, config.max_generation_len, config.seed, result)


if __name__ == "__main__":
    main()
