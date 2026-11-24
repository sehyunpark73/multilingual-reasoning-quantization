from collections.abc import Callable, Iterator
import math
import random
from statistics import mean

import torch
from torch import Tensor
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from transformers import PreTrainedModel, PreTrainedTokenizerBase, get_cosine_schedule_with_warmup

from utils.dataset import NON_EN_LANGUAGES
from utils.model import MODEL_CONFIGS, compute_logits, forward_reasoning_hiddens
from utils.stats import compute_kls
from utils.tokenize import tokenize_en_question, tokenize_prompt

LOG_INTERVAL_STEPS = 16
MAX_REASONING_LEN = 2048


def make_en_question_samples(
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    gsm8k_en_question_insert_reasoning_result: dict,
) -> list[tuple[Tensor, Tensor, Tensor]]:
    reasoning_end_id = tokenizer.convert_tokens_to_ids(MODEL_CONFIGS[model_name]["reasoning_end"])
    samples = []
    for language in NON_EN_LANGUAGES:
        language_result = gsm8k_en_question_insert_reasoning_result[language]
        for prompt, en_question, reasoning_ids in zip(language_result["prompts"], language_result["en_questions"], language_result["reasoning_ids"]):
            if len(reasoning_ids) >= MAX_REASONING_LEN:
                continue
            samples.append((tokenize_prompt(tokenizer, prompt), tokenize_en_question(tokenizer, model_name, en_question), torch.tensor(reasoning_ids + [reasoning_end_id])))
    return samples


def compute_ce_loss(
    model: PreTrainedModel,
    prompt_ids: Tensor,
    text_ids: Tensor,
) -> Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = compute_logits(model, forward_reasoning_hiddens(model, torch.cat([prompt_ids, text_ids]), len(prompt_ids)))
    return cross_entropy(logits, text_ids.to(logits.device), reduction="sum")


def compute_kl_loss(
    model: PreTrainedModel,
    bf16_model: PreTrainedModel,
    prompt_ids: Tensor,
    bf16_prompt_ids: Tensor,
    text_ids: Tensor,
) -> Tensor:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        bf16_logits = compute_logits(bf16_model, forward_reasoning_hiddens(bf16_model, torch.cat([bf16_prompt_ids, text_ids]), len(bf16_prompt_ids)))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = compute_logits(model, forward_reasoning_hiddens(model, torch.cat([prompt_ids, text_ids]), len(prompt_ids)))
    return compute_kls(bf16_logits, logits).sum()


def compute_ce_kl_loss(
    model: PreTrainedModel,
    bf16_model: PreTrainedModel,
    prompt_ids: Tensor,
    en_question_ids: Tensor,
    reasoning_ids: Tensor,
    num_en_question_tokens: int,
    num_reasoning_tokens: int,
) -> Tensor:
    prompt_and_en_question_ids = torch.cat([prompt_ids, en_question_ids])
    ce_loss = compute_ce_loss(model, prompt_ids, en_question_ids) / num_en_question_tokens
    kl_loss = compute_kl_loss(model, bf16_model, prompt_and_en_question_ids, prompt_and_en_question_ids, reasoning_ids) / num_reasoning_tokens
    return ce_loss + kl_loss


def train(
    weights: list[Tensor],
    samples: list,
    compute_losses: Callable[[list], Iterator[Tensor]],
    learning_rate: float,
    num_epochs: int,
    warmup_ratio: float,
    batch_size: int,
    checkpoint_num_trained_samples: tuple[int, ...] = (),
    save_checkpoint: Callable[[int], None] | None = None,
) -> None:
    for weight in weights:
        weight.requires_grad_(True)
    optimizer = AdamW(weights, lr=learning_rate, weight_decay=0)
    num_batches = math.ceil(len(samples) / batch_size)
    num_steps = num_epochs * num_batches
    scheduler = get_cosine_schedule_with_warmup(optimizer, round(warmup_ratio * num_steps), num_steps)

    batch_losses = []
    for epoch in range(num_epochs):
        random.shuffle(samples)
        for batch_index, batch_start in enumerate(range(0, len(samples), batch_size)):
            batch_loss = 0.0
            for sample_loss in compute_losses(samples[batch_start : batch_start + batch_size]):
                sample_loss.backward()
                batch_loss += sample_loss.item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            batch_losses.append(batch_loss)
            step = epoch * num_batches + batch_index + 1
            if step == 1 or step % LOG_INTERVAL_STEPS == 0:
                print(f"epoch {epoch + 1}/{num_epochs} batch {batch_index + 1}/{num_batches} loss {mean(batch_losses):.4f} learning_rate {scheduler.get_last_lr()[0]:.2e}")
                batch_losses = []
            num_trained_samples = epoch * len(samples) + min(batch_start + batch_size, len(samples))
            if num_trained_samples in checkpoint_num_trained_samples:
                save_checkpoint(num_trained_samples)
