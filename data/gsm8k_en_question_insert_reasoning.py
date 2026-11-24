from dataclasses import dataclass
from statistics import mean

from simple_parsing import parse
from transformers import AutoTokenizer

from utils.dataset import NON_EN_LANGUAGES, load_translated_gsm8k
from utils.generate import continue_generation
from utils.hub import download_model, has_result, save_result
from utils.path import RESULT_DIRS, make_model_hub_path
from utils.score import score_generation
from utils.tokenize import tokenize_with_en_questions


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    max_generation_len: int = 16384
    seed: int = 0


def main() -> None:
    config = parse(Config)
    model_hub_path = make_model_hub_path("bf16", config.model_name)

    if has_result(RESULT_DIRS["gsm8k_en_question_insert_reasoning"], model_hub_path, config.max_generation_len, config.seed):
        print(f"{model_hub_path} already generated")
        return

    problems = load_translated_gsm8k()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    input_ids_lists = [tokenize_with_en_questions(tokenizer, config.model_name, problems[language]["prompts"], problems["en"]["questions"]) for language in NON_EN_LANGUAGES]
    input_ids_list_seeds = [config.seed] * len(NON_EN_LANGUAGES)
    generations = continue_generation(download_model(model_hub_path), config.model_name, input_ids_lists, input_ids_list_seeds, config.max_generation_len, is_reasoning_ids_returned=True)

    full_result = {}
    result = {}
    for language, generation in zip(NON_EN_LANGUAGES, generations):
        full_result[language] = score_generation(generation, problems[language]["answers"], "mgsm", language)
        finished_problem_indices = [problem_index for problem_index, response in enumerate(generation["responses"]) if response is not None]
        result[language] = {
            "prompts": [problems[language]["prompts"][problem_index] for problem_index in finished_problem_indices],
            "en_questions": [problems["en"]["questions"][problem_index] for problem_index in finished_problem_indices],
            "reasoning_ids": [generation["reasoning_ids"][problem_index] for problem_index in finished_problem_indices],
            "is_corrects": [full_result[language]["is_corrects"][problem_index] for problem_index in finished_problem_indices],
        }
        print(f"{model_hub_path} {language} accuracy {full_result[language]['accuracy']:.2%} reasoning len {mean(generation['reasoning_lens']):.0f} truncated {mean(generation['is_truncateds']):.2%} finished {len(finished_problem_indices)} of {len(generation['responses'])}")

    save_result(RESULT_DIRS["gsm8k_en_question_insert_reasoning"], model_hub_path, config.max_generation_len, config.seed, result)


if __name__ == "__main__":
    main()
