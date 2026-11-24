import torch
from torch import Tensor
from torch.nn import Linear, Module
from transformers import AutoModelForImageTextToText, PreTrainedModel, PreTrainedTokenizerBase

MODEL_CONFIGS = {
    "Qwen/Qwen3.5-2B": {"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}, "reasoning_start": "", "reasoning_end": "</think>"},
    "google/gemma-4-E2B-it": {"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "presence_penalty": 0.0}, "reasoning_start": "<|channel>", "reasoning_end": "<channel|>"},
}


def load_model(
    model_name: str,
) -> PreTrainedModel:
    model = AutoModelForImageTextToText.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto")
    model.requires_grad_(False)
    return model


def find_linears(
    module: Module,
) -> list[Linear]:
    return [submodule for submodule in module.modules() if isinstance(submodule, Linear)]


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
