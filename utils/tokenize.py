import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from utils.model import MODEL_CONFIGS
from utils.understanding import extract_understanding


def tokenize_prompt(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
) -> Tensor:
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=True, return_tensors="pt")["input_ids"][0]


def tokenize_reasoning(
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    reasoning: str,
) -> Tensor:
    return tokenizer(MODEL_CONFIGS[model_name]["reasoning_start"] + reasoning, add_special_tokens=False, return_tensors="pt")["input_ids"][0]


def tokenize_en_question(
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    en_question: str,
) -> Tensor:
    return tokenize_reasoning(tokenizer, model_name, f"{MODEL_CONFIGS[model_name]['reasoning_header']}{en_question}\n\n")


def tokenize_with_en_questions(
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    prompts: list[str],
    en_questions: list[str],
) -> list[Tensor]:
    input_ids_list = []
    for prompt, en_question in zip(prompts, en_questions):
        input_ids_list.append(torch.cat([tokenize_prompt(tokenizer, prompt), tokenize_en_question(tokenizer, model_name, en_question)]))
    return input_ids_list


def tokenize_with_understandings(
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    prompts: list[str],
    bf16_reasonings: list[str],
) -> tuple[list[int], list[Tensor]]:
    inserted_problem_indices = []
    input_ids_list = []
    for problem_index, reasoning in enumerate(bf16_reasonings):
        understanding = extract_understanding(reasoning)
        if understanding is None:
            continue
        prompt_ids = tokenize_prompt(tokenizer, prompts[problem_index])
        understanding_ids = tokenize_reasoning(tokenizer, model_name, understanding)
        inserted_problem_indices.append(problem_index)
        input_ids_list.append(torch.cat([prompt_ids, understanding_ids]))
    return inserted_problem_indices, input_ids_list
