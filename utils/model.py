import torch
from torch import Tensor
from torch.nn import Linear, Module
from transformers import AutoModelForImageTextToText, PreTrainedModel

MODEL_CONFIGS = {
    "Qwen/Qwen3.5-2B": {"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}, "reasoning_start": "", "reasoning_end": "</think>"},
    "Qwen/Qwen3.5-27B": {"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}, "reasoning_start": "", "reasoning_end": "</think>"},
    "google/gemma-4-E2B-it": {"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "presence_penalty": 0.0}, "reasoning_start": "<|channel>", "reasoning_end": "<channel|>"},
}


class _StopForward(Exception):
    pass


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


def capture_layer_inputs(
    model: PreTrainedModel,
    layer: Module,
    chunks: list[Tensor],
) -> list[tuple[tuple, dict]]:
    layer_inputs = []

    def hook(
        module: Module,
        args: tuple,
        kwargs: dict,
    ) -> None:
        layer_inputs.append((args, kwargs))
        raise _StopForward

    handle = layer.register_forward_pre_hook(hook, with_kwargs=True)
    for chunk in chunks:
        try:
            model(chunk[None, :].to(model.device), use_cache=False)
        except _StopForward:
            pass
    handle.remove()
    return layer_inputs


def forward_reasoning_hiddens(
    model: PreTrainedModel,
    input_ids: Tensor,
    prompt_len: int,
) -> Tensor:
    return model.model(input_ids=input_ids[None, :].to(model.device), use_cache=False).last_hidden_state[0, prompt_len - 1 : -1]


def compute_logits(
    model: PreTrainedModel,
    hiddens: Tensor,
) -> Tensor:
    logits = model.lm_head(hiddens).float()
    softcapping = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
    if softcapping is None:
        return logits
    return torch.tanh(logits / softcapping) * softcapping
