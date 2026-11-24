from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
import random
from typing import Literal, TypeAlias

from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn import Linear, Module
from torch.nn.utils import parametrize
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, set_seed

from utils.dataset import NON_EN_LANGUAGES
from utils.hub import has_model, load_result, save_model
from utils.log import log_time_and_memory
from utils.model import MODEL_CONFIGS, find_linears, load_model
from utils.path import make_model_hub_path
from utils.quantize import quantize_mxfp4
from utils.tokenize import tokenize_prompt
from utils.train import MAX_REASONING_LEN, compute_ce_loss, compute_kl_loss, make_en_question_samples, train

FINETUNE_NAME = "qat"

_Loss: TypeAlias = Literal["ce_kl", "kl"]
_Sample: TypeAlias = tuple[Tensor, Tensor, Tensor]


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    loss: _Loss = "ce_kl"
    gsm8k_en_question_insert_reasoning_hub_path: str = "gsm8k_en_question_insert_reasoning/bf16/google/gemma-4-E2B-it/max_generation_len=16384,seed=0.json"
    gsm8k_reasoning_hub_path: str = "gsm8k_reasoning/bf16/google/gemma-4-E2B-it/max_generation_len=16384,seed=0.json"
    num_samples: int = 20000
    checkpoint_num_trained_samples: tuple[int, ...] = ()
    learning_rate: float = 1e-5
    num_epochs: int = 1
    warmup_ratio: float = 0.05
    batch_size: int = 8
    train_seed: int = 0


class _Mxfp4Ste(Module):
    def forward(
        self,
        weight: Tensor,
    ) -> Tensor:
        with torch.no_grad():
            quantized_weight = quantize_mxfp4(weight)
        return weight + (quantized_weight - weight).detach()


def _make_native_reasoning_samples(
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    gsm8k_reasoning_result: dict,
) -> list[_Sample]:
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


def _compute_losses(
    model: PreTrainedModel,
    bf16_model: PreTrainedModel,
    loss: _Loss,
    batch: list[_Sample],
) -> Iterator[Tensor]:
    num_en_question_tokens = sum(len(en_question_ids) for _, en_question_ids, _ in batch)
    num_reasoning_tokens = sum(len(reasoning_ids) for _, _, reasoning_ids in batch)
    for prompt_ids, en_question_ids, reasoning_ids in batch:
        if loss == "ce_kl":
            prompt_and_en_question_ids = torch.cat([prompt_ids, en_question_ids])
            yield compute_ce_loss(model, prompt_ids, en_question_ids) / num_en_question_tokens + compute_kl_loss(model, bf16_model, prompt_and_en_question_ids, prompt_and_en_question_ids, reasoning_ids) / num_reasoning_tokens
        if loss == "kl":
            yield compute_kl_loss(model, bf16_model, prompt_ids, prompt_ids, reasoning_ids) / num_reasoning_tokens


@torch.no_grad()
def _save_checkpoint(
    model: PreTrainedModel,
    linears: list[Linear],
    weights: list[Tensor],
    model_name: str,
    loss: _Loss,
    num_samples: int,
    num_trained_samples: int,
) -> None:
    fp32_weights = [weight.detach().clone() for weight in weights]
    for linear in linears:
        parametrize.remove_parametrizations(linear, "weight", leave_parametrized=False)
        linear.weight.copy_(quantize_mxfp4(linear.weight))
    model.to(torch.bfloat16)
    save_model(make_model_hub_path(FINETUNE_NAME, model_name, {"loss": loss, "num_samples": num_samples, "num_trained_samples": num_trained_samples}), model)
    model.float()
    for linear, weight, fp32_weight in zip(linears, weights, fp32_weights):
        weight.copy_(fp32_weight)
        parametrize.register_parametrization(linear, "weight", _Mxfp4Ste())


def main() -> None:
    config = parse(Config)
    num_trained_samples = config.num_epochs * config.num_samples
    model_hub_path = make_model_hub_path(FINETUNE_NAME, config.model_name, {"loss": config.loss, "num_samples": config.num_samples, "num_trained_samples": num_trained_samples})

    if has_model(model_hub_path):
        print(f"{model_hub_path} already finetuned")
        return
    if any(checkpoint_num_trained_sample % config.batch_size for checkpoint_num_trained_sample in config.checkpoint_num_trained_samples):
        raise ValueError(f"checkpoint_num_trained_samples must be multiples of batch_size {config.batch_size}")

    set_seed(config.train_seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = load_model(config.model_name)
    model.float()
    bf16_model = load_model(config.model_name)

    if config.loss == "ce_kl":
        gsm8k_en_question_insert_reasoning_result = load_result(config.gsm8k_en_question_insert_reasoning_hub_path)
        all_samples = make_en_question_samples(tokenizer, config.model_name, gsm8k_en_question_insert_reasoning_result)
    if config.loss == "kl":
        gsm8k_reasoning_result = load_result(config.gsm8k_reasoning_hub_path)
        all_samples = _make_native_reasoning_samples(tokenizer, config.model_name, gsm8k_reasoning_result)
    samples = random.sample(all_samples, config.num_samples)
    print(f"{model_hub_path} samples {len(samples)} of {len(all_samples)}")

    with log_time_and_memory(f"{model_hub_path} finetune"):
        linears = find_linears(model.model.language_model.layers)
        for linear in linears:
            parametrize.register_parametrization(linear, "weight", _Mxfp4Ste())
        weights = [linear.parametrizations.weight.original for linear in linears]
        save_checkpoint = partial(_save_checkpoint, model, linears, weights, config.model_name, config.loss, config.num_samples)
        train(weights, samples, partial(_compute_losses, model, bf16_model, config.loss), config.learning_rate, config.num_epochs, config.warmup_ratio, config.batch_size, config.checkpoint_num_trained_samples + (num_trained_samples,), save_checkpoint)


if __name__ == "__main__":
    main()
