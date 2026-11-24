import torch
from torch.nn import Linear, Module
from transformers import AutoModelForImageTextToText, PreTrainedModel


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
