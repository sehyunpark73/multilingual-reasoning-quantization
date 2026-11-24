from collections.abc import Callable
import re
from statistics import mean

from utils.dataset import LANGUAGES, MMLU_PROX_LITE_TEMPLATES, Benchmark

MGSM_ANSWER_PREFIXES = {"en": "Answer", "zh": "答案", "ja": "答え", "sw": "Jibu", "te": "సమాధానం"}
LOOP_TAIL_LEN = 200
LOOP_PATTERN_LEN = 50
MIN_LOOP_COUNT = 5


def _parse_mgsm_prediction(
    response: str,
    language: str,
) -> float | None:
    text = re.split(rf"{MGSM_ANSWER_PREFIXES[language]}\s*[:：]", response)[-1]
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not numbers:
        numbers = re.findall(r"-?\d+(?:\.\d+)?", response.replace(",", ""))
    if not numbers:
        return None
    return float(numbers[-1])


def _parse_mmlu_prox_lite_prediction(
    response: str,
    language: str,
) -> str | None:
    answer_prefix = MMLU_PROX_LITE_TEMPLATES[language]["answer_format"].split("(")[0].strip()
    response = response.replace("*", "")
    letters = re.findall(rf"(?i:{re.escape(answer_prefix)})\s*[:：]?\s*\(?([A-J])\b\)?", response)
    if not letters:
        letters = re.findall(r"\b([A-J])\b", response)
    if not letters:
        return None
    return letters[-1]


def score_generation(
    generation: dict,
    answers: list[int | str],
    benchmark: Benchmark,
    language: str,
) -> dict:
    if benchmark == "mgsm":
        parse_prediction = _parse_mgsm_prediction
    if benchmark == "mmlu_prox_lite":
        parse_prediction = _parse_mmlu_prox_lite_prediction
    predictions = []
    for response in generation["responses"]:
        prediction = None
        if response is not None:
            prediction = parse_prediction(response, language)
        predictions.append(prediction)
    is_corrects = [prediction == answer for prediction, answer in zip(predictions, answers)]
    return {"accuracy": sum(is_corrects) / len(is_corrects), "is_corrects": is_corrects, "predictions": predictions, **generation}


def find_solved_problem_indices(
    results: dict[int, dict],
    language: str,
) -> list[int]:
    is_corrects_by_seed = [result[language]["is_corrects"] for result in results.values()]
    solved_problem_indices = []
    for problem_index in range(len(is_corrects_by_seed[0])):
        is_solved = True
        for is_corrects in is_corrects_by_seed:
            if not is_corrects[problem_index]:
                is_solved = False
        if is_solved:
            solved_problem_indices.append(problem_index)
    return solved_problem_indices


def _compute_mean_accuracy(
    results: dict[int, dict],
    language: str,
) -> float:
    return mean(result[language]["accuracy"] for result in results.values())


def compute_accuracy_losses(
    bf16_results: dict[int, dict],
    all_results: list[dict[int, dict]],
) -> dict[str, list[float]]:
    losses = {}
    for language in LANGUAGES:
        bf16_accuracy = _compute_mean_accuracy(bf16_results, language)
        losses[language] = [bf16_accuracy - _compute_mean_accuracy(results, language) for results in all_results]
    return losses


def compute_accuracy_loss_pairs(
    bf16_results: dict[int, dict],
    all_no_insert_results: list[dict[int, dict]],
    all_insert_results: list[dict[int, dict]],
) -> dict[str, list[tuple[float, float]]]:
    no_insert_losses = compute_accuracy_losses(bf16_results, all_no_insert_results)
    insert_losses = compute_accuracy_losses(bf16_results, all_insert_results)
    return {language: list(zip(no_insert_losses[language], insert_losses[language])) for language in LANGUAGES}


def _has_loop(
    text: str | None,
) -> bool:
    if text is None or len(text) <= LOOP_TAIL_LEN:
        return False
    pattern = text[-LOOP_TAIL_LEN : -LOOP_TAIL_LEN + LOOP_PATTERN_LEN]
    return text[:-LOOP_TAIL_LEN].count(pattern) >= MIN_LOOP_COUNT


def compute_truncated_ratio(
    language_result: dict,
) -> float:
    num_truncateds = 0
    for is_truncated, is_correct in zip(language_result["is_truncateds"], language_result["is_corrects"]):
        if is_truncated and not is_correct:
            num_truncateds += 1
    return num_truncateds / len(language_result["is_truncateds"])


def compute_loop_ratio(
    language_result: dict,
) -> float:
    num_loops = 0
    for reasoning, response in zip(language_result["reasonings"], language_result["responses"]):
        if _has_loop(reasoning) or _has_loop(response):
            num_loops += 1
    return num_loops / len(language_result["reasonings"])


def compute_mean_values(
    all_results: list[dict[int, dict]],
    compute_value: Callable[[dict], float],
) -> dict[str, list[float]]:
    values = {}
    for language in LANGUAGES:
        values[language] = [mean(compute_value(result[language]) for result in results.values()) for results in all_results]
    return values


def compute_subset_accuracy(
    is_corrects: list[bool],
    result_problem_indices: list[int],
    problem_indices: set[int],
) -> float:
    subset_is_corrects = [is_correct for problem_index, is_correct in zip(result_problem_indices, is_corrects) if problem_index in problem_indices]
    return sum(subset_is_corrects) / len(subset_is_corrects)


def compute_insert_losses(
    bf16_results: dict[int, dict],
    no_insert_results: dict[int, dict],
    insert_results: dict[int, dict],
    language: str,
    problem_indices_by_seed: dict[int, set[int]],
) -> tuple[float, float]:
    bf16_accuracies = []
    no_insert_accuracies = []
    insert_accuracies = []
    for seed, problem_indices in problem_indices_by_seed.items():
        all_problem_indices = list(range(len(bf16_results[seed][language]["is_corrects"])))
        bf16_accuracies.append(compute_subset_accuracy(bf16_results[seed][language]["is_corrects"], all_problem_indices, problem_indices))
        no_insert_accuracies.append(compute_subset_accuracy(no_insert_results[seed][language]["is_corrects"], all_problem_indices, problem_indices))
        insert_accuracies.append(compute_subset_accuracy(insert_results[seed][language]["is_corrects"], insert_results[seed][language]["inserted_problem_indices"], problem_indices))
    return mean(bf16_accuracies) - mean(no_insert_accuracies), mean(bf16_accuracies) - mean(insert_accuracies)


def find_common_inserted_problem_indices(
    native_results: dict[int, dict],
    en_results: dict[int, dict],
    language: str,
) -> dict[int, set[int]]:
    problem_indices_by_seed = {}
    for seed, native_result in native_results.items():
        native_problem_indices = set(native_result[language]["inserted_problem_indices"])
        en_problem_indices = set(en_results[seed][language]["inserted_problem_indices"])
        problem_indices_by_seed[seed] = native_problem_indices & en_problem_indices
    return problem_indices_by_seed
