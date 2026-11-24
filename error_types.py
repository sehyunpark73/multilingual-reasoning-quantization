from dataclasses import dataclass
import re

from simple_parsing import parse

from utils.dataset import LANGUAGES, load_mgsm
from utils.generate import generate
from utils.hub import load_bf16_results, load_results, save_result
from utils.path import extract_model_name, make_loss_result_dir
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
    model_hub_path: str = "mxfp4/google/gemma-4-E2B-it"
    judge_model_name: str = "Qwen/Qwen3.5-27B"
    judge_max_generation_len: int = 16384
    max_generation_len: int = 16384
    seeds: tuple[int, ...] = (0, 1, 2)
    bf16_seeds: tuple[int, ...] = (0, 1, 2)


def _find_wrong_problem_indices(
    bf16_results: dict[int, dict],
    quantized_result: dict,
    language: str,
) -> list[int]:
    return [problem_index for problem_index in find_solved_problem_indices(bf16_results, language) if not quantized_result[language]["is_corrects"][problem_index]]


def _make_judge_prompts(
    problems: dict[str, dict[str, list]],
    language: str,
    wrong_problem_indices: list[int],
    bf16_reasonings: list[str],
    quantized_reasonings: list[str],
) -> list[str]:
    judge_prompts = []
    for problem_index in wrong_problem_indices:
        numbered_lines = [f"{line_index + 1}: {line}" for line_index, line in enumerate(quantized_reasonings[problem_index].split("\n"))]
        judge_prompts.append(JUDGE_PROMPT.format(
            en_question=problems["en"]["questions"][problem_index],
            question=problems[language]["questions"][problem_index],
            answer=problems[language]["answers"][problem_index],
            bf16_reasoning=bf16_reasonings[problem_index],
            numbered_reasoning="\n".join(numbered_lines),
        ))
    return judge_prompts


def _parse_judgments(
    responses: list[str | None],
) -> tuple[list[int | None], list[str | None]]:
    error_lines = []
    error_types = []
    for response in responses:
        line_matches = re.findall(r"line:\s*(\d+)", response or "")
        type_matches = re.findall(rf"type:\s*({'|'.join(ERROR_TYPES)})\b", response or "")
        error_lines.append(int(line_matches[-1]) if line_matches else None)
        error_types.append(type_matches[-1] if type_matches else None)
    return error_lines, error_types


def main() -> None:
    config = parse(Config)
    model_name = extract_model_name(config.model_hub_path)

    results = load_results(RESULT_DIR, config.model_hub_path, config.max_generation_len, config.seeds)
    missing_seeds = [seed for seed, result in results.items() if not result]
    if not missing_seeds:
        print(f"{config.model_hub_path} already evaluated")
        return

    problems = load_mgsm()
    bf16_results = load_bf16_results("mgsm", model_name, config.max_generation_len, config.bf16_seeds)
    quantized_results = load_results(make_loss_result_dir("mgsm"), config.model_hub_path, config.max_generation_len, missing_seeds)

    missing_seed_languages = [(seed, language) for seed in missing_seeds for language in LANGUAGES]
    wrong_problem_indices = {}
    judge_prompt_lists = []
    for seed, language in missing_seed_languages:
        wrong_problem_indices[seed, language] = _find_wrong_problem_indices(bf16_results, quantized_results[seed], language)
        judge_prompt_lists.append(_make_judge_prompts(problems, language, wrong_problem_indices[seed, language], bf16_results[seed][language]["reasonings"], quantized_results[seed][language]["reasonings"]))
        print(f"{config.model_hub_path} seed {seed} {language} wrong {len(wrong_problem_indices[seed, language])}")

    judge_prompt_list_seeds = [seed for seed, _ in missing_seed_languages]
    generations = generate(config.judge_model_name, config.judge_model_name, judge_prompt_lists, judge_prompt_list_seeds, config.judge_max_generation_len, JUDGE_MAX_NUM_SEQUENCES)

    for (seed, language), generation in zip(missing_seed_languages, generations):
        error_lines, error_types = _parse_judgments(generation["responses"])
        results[seed][language] = {"wrong_problem_indices": wrong_problem_indices[seed, language], "error_lines": error_lines, "error_types": error_types, **generation}
        print(f"{config.model_hub_path} seed {seed} {language} judged {len(error_types)} unparsed {error_types.count(None)}")

    for seed in missing_seeds:
        save_result(RESULT_DIR, config.model_hub_path, config.max_generation_len, seed, results[seed])


if __name__ == "__main__":
    main()
