from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
import random
from typing import Literal, TypeAlias

from simple_parsing import parse
import torch
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedModel, set_seed

from utils.hub import has_model, load_result, save_model
from utils.log import log_time_and_memory
from utils.model import find_linears, load_model
from utils.path import make_finetune_model_hub_path
from utils.train import compute_ce_loss, compute_kl_loss, make_en_question_samples, train

FINETUNE_NAME = "question_finetune"

_Loss: TypeAlias = Literal["ce", "ce_kl"]
_Sample: TypeAlias = tuple[Tensor, Tensor, Tensor]


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    loss: _Loss = "ce_kl"
    gsm8k_en_question_insert_reasoning_hub_path: str = "gsm8k_en_question_insert_reasoning/bf16/google/gemma-4-E2B-it/max_generation_len=16384,seed=0.json"
    num_samples: int = 20000
    checkpoint_num_trained_samples: tuple[int, ...] = (800, 4000)
    learning_rate: float = 1e-5
    num_epochs: int = 1
    warmup_ratio: float = 0.05
    batch_size: int = 8
    train_seed: int = 0


def _compute_losses(
    model: PreTrainedModel,
    bf16_model: PreTrainedModel | None,
    loss: _Loss,
    batch: list[_Sample],
) -> Iterator[Tensor]:
    num_en_question_tokens = sum(len(en_question_ids) for _, en_question_ids, _ in batch)
    num_reasoning_tokens = sum(len(reasoning_ids) for _, _, reasoning_ids in batch)
    for prompt_ids, en_question_ids, reasoning_ids in batch:
        if loss == "ce":
            yield compute_ce_loss(model, prompt_ids, en_question_ids) / num_en_question_tokens
        if loss == "ce_kl":
            prompt_and_en_question_ids = torch.cat([prompt_ids, en_question_ids])
            yield compute_ce_loss(model, prompt_ids, en_question_ids) / num_en_question_tokens + compute_kl_loss(model, bf16_model, prompt_and_en_question_ids, prompt_and_en_question_ids, reasoning_ids) / num_reasoning_tokens


@torch.no_grad()
def _save_checkpoint(
    model: PreTrainedModel,
    weights: list[Tensor],
    model_name: str,
    loss: _Loss,
    num_samples: int,
    num_trained_samples: int,
) -> None:
    fp32_weights = [weight.detach().clone() for weight in weights]
    model.to(torch.bfloat16)
    save_model(make_finetune_model_hub_path(FINETUNE_NAME, model_name, {"loss": loss, "num_samples": num_samples, "num_trained_samples": num_trained_samples}), model)
    model.float()
    for weight, fp32_weight in zip(weights, fp32_weights):
        weight.copy_(fp32_weight)


def main() -> None:
    config = parse(Config)
    num_trained_samples = config.num_epochs * config.num_samples
    model_hub_path = make_finetune_model_hub_path(FINETUNE_NAME, config.model_name, {"loss": config.loss, "num_samples": config.num_samples, "num_trained_samples": num_trained_samples})

    if has_model(model_hub_path):
        print(f"{model_hub_path} already finetuned")
        return
    if any(checkpoint_num_trained_sample % config.batch_size for checkpoint_num_trained_sample in config.checkpoint_num_trained_samples):
        raise ValueError(f"checkpoint_num_trained_samples must be multiples of batch_size {config.batch_size}")

    set_seed(config.train_seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = load_model(config.model_name)
    model.float()
    bf16_model = None
    if config.loss == "ce_kl":
        bf16_model = load_model(config.model_name)

    gsm8k_en_question_insert_reasoning_result = load_result(config.gsm8k_en_question_insert_reasoning_hub_path)
    all_samples = make_en_question_samples(tokenizer, config.model_name, gsm8k_en_question_insert_reasoning_result)
    samples = random.sample(all_samples, config.num_samples)
    print(f"{model_hub_path} samples {len(samples)} of {len(all_samples)}")

    with log_time_and_memory(f"{model_hub_path} finetune"):
        weights = [linear.weight for linear in find_linears(model.model.language_model.layers)]
        save_checkpoint = partial(_save_checkpoint, model, weights, config.model_name, config.loss, config.num_samples)
        train(weights, samples, partial(_compute_losses, model, bf16_model, config.loss), config.learning_rate, config.num_epochs, config.warmup_ratio, config.batch_size, config.checkpoint_num_trained_samples + (num_trained_samples,), save_checkpoint)


if __name__ == "__main__":
    main()
