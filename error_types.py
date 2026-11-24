from dataclasses import dataclass
import re
from typing import Literal

from simple_parsing import parse

from utils.dataset import LANGUAGES, load_mgsm
from utils.generate import generate
from utils.hub import load_results, make_hub_path, save_result
from utils.score import find_solved_problem_indices

RESULT_DIR = "error_types/mgsm"
ERROR_TYPES = ["fact", "relation", "target", "doubt", "arithmetic", "other"]
JUDGE_PROMPT = """A small model solved a math word problem and got it wrong. Find the first line where its reasoning goes wrong, and say what kind of error it is.

Problem in English, which is the true meaning:
{en_question}

Problem as the model saw it:
{question}

Correct answer: {answer}

Reasoning of a stronger model that got it right:
<correct>
{bf16_reasoning}
</correct>

Reasoning of the small model that got it wrong, with line numbers:
<wrong>
{numbered_reasoning}
</wrong>

Error types:
- fact: states a number, unit, word or entity of the problem wrong (e.g. "runs 15 miles" becomes "runs 15 minutes")
- relation: states a relation between quantities wrong (e.g. "4 more cats than dogs" becomes "4 cats")
- target: solves for a different quantity than the one asked (e.g. problem asks how many are left, reasoning gives how many were used)
- doubt: calls a given condition missing, ambiguous or irrelevant, and drops it (e.g. "the price per box is not given", though the problem gives it)
- arithmetic: sets up the problem right, but computes wrong (e.g. "7 x 8 = 54")
- other: none of the above

End your answer with exactly these two lines (e.g. for a wrong relation on line 7):
line: 7
type: relation"""
JUDGE_MAX_NUM_SEQUENCES = 256


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    method: Literal["gptq", "awq", "nf4", "mxfp4"] = "mxfp4"
    calibration_languages: tuple[Literal["en", "zh", "ja", "sw", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_chunks: int = 128
    judge_model_name: str = "Qwen/Qwen3.5-27B"
    judge_max_generation_len: int = 16384
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)
    bf16_seeds: tuple[int, ...] = (0, 1, 2)


def _make_judge_prompt(
    en_question: str,
    question: str,
    answer: int,
    bf16_reasoning: str,
    reasoning: str,
) -> str:
    numbered_lines = [f"{line_index + 1}: {line}" for line_index, line in enumerate(reasoning.split("\n"))]
    return JUDGE_PROMPT.format(en_question=en_question, question=question, answer=answer, bf16_reasoning=bf16_reasoning, numbered_reasoning="\n".join(numbered_lines))


def _parse_judgment(
    response: str | None,
) -> tuple[int | None, str | None]:
    line_matches = re.findall(r"line:\s*(\d+)", response or "")
    type_matches = re.findall(rf"type:\s*({'|'.join(ERROR_TYPES)})\b", response or "")
    return int(line_matches[-1]) if line_matches else None, type_matches[-1] if type_matches else None


def main() -> None:
    config = parse(Config)
    hub_path = make_hub_path(config.method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_chunks)

    results = load_results(RESULT_DIR, hub_path, config.max_generation_len, config.seeds)
    missing_seeds = [seed for seed, result in results.items() if not result]
    if not missing_seeds:
        print(f"{hub_path} already evaluated")
        return

    problems = load_mgsm()
    bf16_results = load_results("loss/mgsm", make_hub_path("bf16", config.model_name), config.max_generation_len, config.bf16_seeds)
    quantized_results = load_results("loss/mgsm", hub_path, config.max_generation_len, tuple(missing_seeds))
    seed_languages = [(seed, language) for seed in missing_seeds for language in LANGUAGES]
    wrong_problem_indices = {}
    judge_prompt_lists = []
    for seed, language in seed_languages:
        solved_problem_indices = find_solved_problem_indices(bf16_results, language)
        wrong_problem_indices[seed, language] = [problem_index for problem_index in solved_problem_indices if not quantized_results[seed][language]["corrects"][problem_index]]
        judge_prompts = []
        for problem_index in wrong_problem_indices[seed, language]:
            judge_prompts.append(_make_judge_prompt(
                problems["en"]["questions"][problem_index],
                problems[language]["questions"][problem_index],
                problems[language]["answers"][problem_index],
                bf16_results[seed][language]["reasonings"][problem_index],
                quantized_results[seed][language]["reasonings"][problem_index],
            ))
        judge_prompt_lists.append(judge_prompts)
        print(f"{hub_path} seed {seed} {language} wrong {len(judge_prompts)} of {len(solved_problem_indices)} solved")

    judge_prompt_list_seeds = [seed for seed, _ in seed_languages]
    generations = generate(config.judge_model_name, config.judge_model_name, judge_prompt_lists, judge_prompt_list_seeds, config.judge_max_generation_len, JUDGE_MAX_NUM_SEQUENCES)

    for (seed, language), generation in zip(seed_languages, generations):
        error_lines = []
        error_types = []
        for response in generation["responses"]:
            error_line, error_type = _parse_judgment(response)
            error_lines.append(error_line)
            error_types.append(error_type)
        results[seed][language] = {"wrong_problem_indices": wrong_problem_indices[seed, language], "error_lines": error_lines, "error_types": error_types, **generation}
        print(f"{hub_path} seed {seed} {language} judged {len(error_types)} unparsed {error_types.count(None)}")

    for seed in missing_seeds:
        save_result(RESULT_DIR, hub_path, config.max_generation_len, seed, results[seed])


if __name__ == "__main__":
    main()
