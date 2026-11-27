from collections.abc import Callable, Iterator
import math
import random
from statistics import mean
import time
from typing import TypeAlias

import torch
from torch import Tensor
from torch.nn import Linear
from torch.nn.functional import cross_entropy
from torch.nn.utils import parametrize
from torch.optim import AdamW
from transformers import PreTrainedModel, PreTrainedTokenizerBase, get_cosine_schedule_with_warmup

from utils.dataset import NON_EN_LANGUAGES
from utils.hub import save_model
from utils.log import format_time_and_memory
from utils.model import MODEL_CONFIGS, compute_logits, forward_reasoning_hiddens
from utils.path import make_model_hub_path
from utils.quantize import Fp4Method, QuantizeSte, quantize_fp4
from utils.stats import compute_kls
from utils.tokenize import tokenize_en_question, tokenize_prompt

LOG_INTERVAL_STEPS = 16
MAX_REASONING_LEN = 2048

Sample: TypeAlias = tuple[Tensor, Tensor, Tensor]


def make_en_question_samples(
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    gsm8k_en_question_insert_reasoning_result: dict,
) -> list[Sample]:
    reasoning_end_id = tokenizer.convert_tokens_to_ids(MODEL_CONFIGS[model_name]["reasoning_end"])
    samples = []
    for language in NON_EN_LANGUAGES:
        language_result = gsm8k_en_question_insert_reasoning_result[language]
        for prompt, en_question, reasoning_ids in zip(language_result["prompts"], language_result["en_questions"], language_result["reasoning_ids"]):
            if len(reasoning_ids) >= MAX_REASONING_LEN:
                continue
            samples.append((tokenize_prompt(tokenizer, prompt), tokenize_en_question(tokenizer, model_name, en_question), torch.tensor(reasoning_ids + [reasoning_end_id])))
    return samples


def make_native_reasoning_samples(
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    gsm8k_reasoning_result: dict,
) -> list[Sample]:
    reasoning_end_id = tokenizer.convert_tokens_to_ids(MODEL_CONFIGS[model_name]["reasoning_end"])
    empty_en_question_ids = torch.tensor([], dtype=torch.long)
    samples = []
    for language in NON_EN_LANGUAGES:
        language_result = gsm8k_reasoning_result[language]
        for prompt, reasoning_ids in zip(language_result["prompts"], language_result["reasoning_ids"]):
            if len(reasoning_ids) >= MAX_REASONING_LEN:
                continue
            samples.append((tokenize_prompt(tokenizer, prompt), empty_en_question_ids, torch.tensor(reasoning_ids + [reasoning_end_id])))
    return samples


def compute_ce_loss(
    model: PreTrainedModel,
    prompt_ids: Tensor,
    text_ids: Tensor,
) -> Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = compute_logits(model, forward_reasoning_hiddens(model, torch.cat([prompt_ids, text_ids]), len(prompt_ids)))
    return cross_entropy(logits, text_ids.to(logits.device), reduction="sum")


def _compute_kl_sum(
    bf16_logits: Tensor,
    logits: Tensor,
    max_token_kl: float | None,
) -> Tensor:
    kls = compute_kls(bf16_logits, logits)
    if max_token_kl is not None:
        kls = kls.clamp(max=max_token_kl)
    return kls.sum()


def compute_kl_loss(
    model: PreTrainedModel,
    bf16_model: PreTrainedModel,
    prompt_ids: Tensor,
    bf16_prompt_ids: Tensor,
    text_ids: Tensor,
    max_token_kl: float | None = None,
) -> Tensor:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        bf16_logits = compute_logits(bf16_model, forward_reasoning_hiddens(bf16_model, torch.cat([bf16_prompt_ids, text_ids]), len(bf16_prompt_ids)))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = compute_logits(model, forward_reasoning_hiddens(model, torch.cat([prompt_ids, text_ids]), len(prompt_ids)))
    return _compute_kl_sum(bf16_logits, logits, max_token_kl)


def compute_ce_kl_loss(
    model: PreTrainedModel,
    bf16_model: PreTrainedModel,
    prompt_ids: Tensor,
    en_question_ids: Tensor,
    reasoning_ids: Tensor,
    num_en_question_tokens: int,
    num_reasoning_tokens: int,
    max_token_kl: float | None = None,
) -> Tensor:
    input_ids = torch.cat([prompt_ids, en_question_ids, reasoning_ids])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        bf16_logits = compute_logits(bf16_model, forward_reasoning_hiddens(bf16_model, input_ids, len(prompt_ids) + len(en_question_ids)))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = compute_logits(model, forward_reasoning_hiddens(model, input_ids, len(prompt_ids)))
    ce_loss = cross_entropy(logits[: len(en_question_ids)], en_question_ids.to(logits.device), reduction="sum") / num_en_question_tokens
    kl_loss = _compute_kl_sum(bf16_logits, logits[len(en_question_ids) :], max_token_kl) / num_reasoning_tokens
    return ce_loss + kl_loss


def train(
    weights: list[Tensor],
    samples: list[Sample],
    compute_losses: Callable[[list[Sample]], Iterator[Tensor]],
    learning_rate: float,
    num_epochs: int,
    warmup_ratio: float,
    batch_size: int,
    checkpoint_num_trained_samples: tuple[int, ...] = (),
    save_checkpoint: Callable[[int], None] | None = None,
    generate_batches: Callable[[list[list[Sample]]], list[list[Sample]]] | None = None,
    num_batches_per_generation: int = 1,
) -> None:
    for weight in weights:
        weight.requires_grad_(True)
    optimizer = AdamW(weights, lr=learning_rate, weight_decay=0)
    num_batches = math.ceil(len(samples) / batch_size)
    num_steps = num_epochs * num_batches
    scheduler = get_cosine_schedule_with_warmup(optimizer, round(warmup_ratio * num_steps), num_steps)

    batch_losses = []
    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(num_epochs):
        random.shuffle(samples)
        batches = [samples[batch_start : batch_start + batch_size] for batch_start in range(0, len(samples), batch_size)]
        for batch_index in range(num_batches):
            if generate_batches is not None and batch_index % num_batches_per_generation == 0:
                batches[batch_index : batch_index + num_batches_per_generation] = generate_batches(batches[batch_index : batch_index + num_batches_per_generation])
            batch_loss = 0.0
            for sample_loss in compute_losses(batches[batch_index]):
                sample_loss.backward()
                batch_loss += sample_loss.item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            batch_losses.append(batch_loss)
            step = epoch * num_batches + batch_index + 1
            if step == 1 or step % LOG_INTERVAL_STEPS == 0:
                print(f"epoch {epoch + 1}/{num_epochs} batch {batch_index + 1}/{num_batches} loss {mean(batch_losses):.4f} learning_rate {scheduler.get_last_lr()[0]:.2e} {format_time_and_memory(start_time)}")
                batch_losses = []
                start_time = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
            num_trained_samples = epoch * len(samples) + batch_index * batch_size + len(batches[batch_index])
            if num_trained_samples in checkpoint_num_trained_samples:
                save_checkpoint(num_trained_samples)


@torch.no_grad()
def save_qat_checkpoint(
    model: PreTrainedModel,
    linears: list[Linear],
    weights: list[Tensor],
    method: Fp4Method,
    finetune_name: str,
    model_name: str,
    settings: dict,
    num_trained_samples: int,
) -> None:
    fp32_weights = [weight.detach().clone() for weight in weights]
    for linear in linears:
        parametrize.remove_parametrizations(linear, "weight", leave_parametrized=False)
        linear.weight.copy_(quantize_fp4(linear.weight, method))
    model.to(torch.bfloat16)
    save_model(make_model_hub_path(finetune_name, model_name, settings | {"num_trained_samples": num_trained_samples}), model)
    model.float()
    for linear, weight, fp32_weight in zip(linears, weights, fp32_weights):
        weight.copy_(fp32_weight)
        parametrize.register_parametrization(linear, "weight", QuantizeSte(method))
