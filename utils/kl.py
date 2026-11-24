import torch
from torch import Tensor
from torch.nn.functional import log_softmax
from transformers import PreTrainedModel


def _compute_reasoning_hiddens(
    model: PreTrainedModel,
    input_ids: Tensor,
    prompt_len: int,
) -> Tensor:
    return model.model(input_ids=input_ids[None, :].to(model.device), use_cache=False).last_hidden_state[0, prompt_len - 1 : -1]


def _compute_logprobs(
    model: PreTrainedModel,
    hiddens: Tensor,
) -> Tensor:
    logits = model.lm_head(hiddens).float()
    softcapping = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
    if softcapping is not None:
        logits = torch.tanh(logits / softcapping) * softcapping
    return log_softmax(logits, dim=-1)


def compute_reasoning_kls(
    bf16_model: PreTrainedModel,
    quantized_model: PreTrainedModel,
    input_ids: Tensor,
    prompt_len: int,
    logit_batch_len: int,
) -> Tensor:
    bf16_hiddens = _compute_reasoning_hiddens(bf16_model, input_ids, prompt_len)
    quantized_hiddens = _compute_reasoning_hiddens(quantized_model, input_ids, prompt_len)
    kls = []
    for batch_start in range(0, len(bf16_hiddens), logit_batch_len):
        bf16_logprobs = _compute_logprobs(bf16_model, bf16_hiddens[batch_start : batch_start + logit_batch_len])
        quantized_logprobs = _compute_logprobs(quantized_model, quantized_hiddens[batch_start : batch_start + logit_batch_len])
        kls.append((bf16_logprobs.exp() * (bf16_logprobs - quantized_logprobs)).sum(dim=-1).cpu())
    return torch.cat(kls)
