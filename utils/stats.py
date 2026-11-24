import torch
from torch import Tensor
from torch.nn import Linear, Module
from torch.nn.functional import log_softmax
from transformers import PreTrainedModel

from utils.model import compute_logits, forward_reasoning_hiddens


def compute_abs_mean_and_hessian(
    layer: Module,
    linear: Linear,
    layer_inputs: list[tuple[tuple, dict]],
) -> tuple[Tensor, Tensor]:
    abs_sum = torch.zeros(linear.in_features, device=linear.weight.device)
    hessian = torch.zeros(linear.in_features, linear.in_features, device=linear.weight.device)
    num_tokens = 0

    def hook(
        module: Module,
        args: tuple[Tensor, ...],
    ) -> None:
        nonlocal num_tokens
        activations = args[0].reshape(-1, args[0].shape[-1]).float()
        abs_sum.add_(activations.abs().sum(dim=0))
        hessian.add_(activations.T @ activations)
        num_tokens += activations.shape[0]

    handle = linear.register_forward_pre_hook(hook)
    for layer_args, layer_kwargs in layer_inputs:
        layer(*layer_args, **layer_kwargs)
    handle.remove()
    return abs_sum / num_tokens, hessian


def compute_kls(
    bf16_logits: Tensor,
    logits: Tensor,
) -> Tensor:
    bf16_logprobs = log_softmax(bf16_logits, dim=-1)
    logprobs = log_softmax(logits, dim=-1)
    return (bf16_logprobs.exp() * (bf16_logprobs - logprobs)).sum(dim=-1)


def compute_reasoning_kls(
    bf16_model: PreTrainedModel,
    quantized_model: PreTrainedModel,
    input_ids: Tensor,
    prompt_len: int,
    logit_batch_len: int,
) -> Tensor:
    bf16_hiddens = forward_reasoning_hiddens(bf16_model, input_ids, prompt_len)
    quantized_hiddens = forward_reasoning_hiddens(quantized_model, input_ids, prompt_len)
    kls = []
    for batch_start in range(0, len(bf16_hiddens), logit_batch_len):
        bf16_logits = compute_logits(bf16_model, bf16_hiddens[batch_start : batch_start + logit_batch_len])
        quantized_logits = compute_logits(quantized_model, quantized_hiddens[batch_start : batch_start + logit_batch_len])
        kls.append(compute_kls(bf16_logits, quantized_logits).cpu())
    return torch.cat(kls)
