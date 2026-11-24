import re

from utils.dataset import MMLU_PROX_LITE_TEMPLATES, Benchmark

MGSM_ANSWER_PREFIXES = {"en": "Answer", "zh": "答案", "ja": "答え", "sw": "Jibu", "te": "సమాధానం"}


def _parse_mgsm_prediction(
    response: str,
    language: str,
) -> float | None:
    text = re.split(rf"{MGSM_ANSWER_PREFIXES[language]}\s*[:：]", response)[-1]
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", "")) or re.findall(r"-?\d+(?:\.\d+)?", response.replace(",", ""))
    return float(numbers[-1]) if numbers else None


def _parse_mmlu_prox_lite_prediction(
    response: str,
    language: str,
) -> str | None:
    answer_prefix = MMLU_PROX_LITE_TEMPLATES[language]["answer_format"].split("(")[0].strip()
    response = response.replace("*", "")
    letters = re.findall(rf"(?i:{re.escape(answer_prefix)})\s*[:：]?\s*\(?([A-J])\b\)?", response)
    letters = letters or re.findall(r"\b([A-J])\b", response)
    return letters[-1] if letters else None


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
    predictions = [None if response is None else parse_prediction(response, language) for response in generation["responses"]]
    is_corrects = [prediction == answer for prediction, answer in zip(predictions, answers)]
    return {"accuracy": sum(is_corrects) / len(is_corrects), "is_corrects": is_corrects, "predictions": predictions, **generation}


def find_solved_problem_indices(
    results: dict[int, dict],
    language: str,
) -> list[int]:
    is_corrects_by_seed = [result[language]["is_corrects"] for result in results.values()]
    return [problem_index for problem_index, is_corrects in enumerate(zip(*is_corrects_by_seed)) if all(is_corrects)]
