import re

from utils.dataset import MMLU_PROX_LITE_TEMPLATES, Benchmark

MGSM_ANSWER_PREFIXES = {"en": "Answer", "zh": "答案", "ja": "答え", "sw": "Jibu", "te": "సమాధానం"}


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
