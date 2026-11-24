import torch
from torch.nn import Linear, Module
from transformers import AutoModelForImageTextToText, PreTrainedModel

MODEL_CONFIGS = {
    "Qwen/Qwen3.5-2B": {"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}, "reasoning_start": "", "reasoning_end": "</think>"},
    "Qwen/Qwen3.5-27B": {"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}, "reasoning_start": "", "reasoning_end": "</think>"},
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
