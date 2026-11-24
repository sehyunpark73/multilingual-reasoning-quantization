from dataclasses import dataclass
from statistics import mean

from simple_parsing import parse
import torch
from transformers import AutoTokenizer

from utils.dataset import LANGUAGES, load_translated_gsm8k
from utils.generate import continue_generation
from utils.hub import has_result, save_result
from utils.path import make_model_hub_path
from utils.score import score_generation
from utils.tokenize import tokenize_en_question, tokenize_prompt

RESULT_DIR = "gsm8k_en_question_insert_reasoning"
INSERTED_LANGUAGES = [language for language in LANGUAGES if language != "en"]


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
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    en_question_ids_list = [tokenize_en_question(tokenizer, config.model_name, en_question) for en_question in problems["en"]["questions"]]
    input_ids_lists = []
    for language in INSERTED_LANGUAGES:
        input_ids_lists.append([torch.cat([tokenize_prompt(tokenizer, prompt), en_question_ids]) for prompt, en_question_ids in zip(problems[language]["prompts"], en_question_ids_list)])
    input_ids_list_seeds = [config.seed] * len(INSERTED_LANGUAGES)
    generations = continue_generation(config.model_name, config.model_name, input_ids_lists, input_ids_list_seeds, config.max_generation_len, is_reasoning_ids_returned=True)

    result = {}
    for language, generation in zip(INSERTED_LANGUAGES, generations):
        scored_generation = score_generation(generation, problems[language]["answers"], "mgsm", language)
        finished_problem_indices = [problem_index for problem_index, response in enumerate(generation["responses"]) if response is not None]
        result[language] = {
            "prompts": [problems[language]["prompts"][problem_index] for problem_index in finished_problem_indices],
            "en_questions": [problems["en"]["questions"][problem_index] for problem_index in finished_problem_indices],
            "reasoning_ids": [generation["reasoning_ids"][problem_index] for problem_index in finished_problem_indices],
            "is_corrects": [scored_generation["is_corrects"][problem_index] for problem_index in finished_problem_indices],
        }
        print(f"{model_hub_path} {language} accuracy {scored_generation['accuracy']:.2%} reasoning len {mean(generation['reasoning_lens']):.0f} truncated {mean(generation['is_truncateds']):.2%} finished {len(finished_problem_indices)} of {len(generation['responses'])}")

    save_result(RESULT_DIR, model_hub_path, config.max_generation_len, config.seed, result)


if __name__ == "__main__":
    main()
