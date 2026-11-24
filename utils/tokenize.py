from torch import Tensor
from transformers import PreTrainedTokenizerBase

from utils.model import MODEL_CONFIGS


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
