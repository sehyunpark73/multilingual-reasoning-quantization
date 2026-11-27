from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
import random
from typing import Literal, TypeAlias

from simple_parsing import parse
from torch import Tensor
from torch.nn.utils import parametrize
from transformers import AutoTokenizer, PreTrainedModel, set_seed

from utils.hub import has_model, load_result
from utils.log import log_time
from utils.model import find_linears, load_model
from utils.path import make_model_hub_path
from utils.quantize import Fp4Method, QuantizeSte
from utils.train import Sample, compute_ce_kl_loss, compute_kl_loss, make_en_question_samples, make_native_reasoning_samples, save_qat_checkpoint, train

FINETUNE_NAME = "qat"
METHOD: Fp4Method = "mxfp4"

_Loss: TypeAlias = Literal["ce_kl", "kl"]


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


def _compute_losses(
    model: PreTrainedModel,
    bf16_model: PreTrainedModel,
    loss: _Loss,
    batch: list[Sample],
) -> Iterator[Tensor]:
    num_en_question_tokens = sum(len(en_question_ids) for _, en_question_ids, _ in batch)
    num_reasoning_tokens = sum(len(reasoning_ids) for _, _, reasoning_ids in batch)
    for prompt_ids, en_question_ids, reasoning_ids in batch:
        if loss == "ce_kl":
            yield compute_ce_kl_loss(model, bf16_model, prompt_ids, en_question_ids, reasoning_ids, num_en_question_tokens, num_reasoning_tokens)
        if loss == "kl":
            yield compute_kl_loss(model, bf16_model, prompt_ids, prompt_ids, reasoning_ids) / num_reasoning_tokens


def main() -> None:
    config = parse(Config)
    num_trained_samples = config.num_epochs * config.num_samples
    model_hub_path = make_model_hub_path(FINETUNE_NAME, config.model_name, {"loss": config.loss, "num_samples": config.num_samples, "num_trained_samples": num_trained_samples})

    if has_model(model_hub_path):
        print(f"{model_hub_path} already finetuned")
        return
    for checkpoint_num_trained_samples in config.checkpoint_num_trained_samples:
        if checkpoint_num_trained_samples % config.batch_size != 0:
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
        all_samples = make_native_reasoning_samples(tokenizer, config.model_name, gsm8k_reasoning_result)
    samples = random.sample(all_samples, config.num_samples)
    print(f"{model_hub_path} samples {len(samples)} of {len(all_samples)}")

    with log_time(f"{model_hub_path} finetune"):
        linears = find_linears(model.model.language_model.layers)
        for linear in linears:
            parametrize.register_parametrization(linear, "weight", QuantizeSte(METHOD))
        weights = [linear.parametrizations.weight.original for linear in linears]
        save_checkpoint = partial(save_qat_checkpoint, model, linears, weights, METHOD, FINETUNE_NAME, config.model_name, {"loss": config.loss, "num_samples": config.num_samples})
        train(weights, samples, partial(_compute_losses, model, bf16_model, config.loss), config.learning_rate, config.num_epochs, config.warmup_ratio, config.batch_size, config.checkpoint_num_trained_samples + (num_trained_samples,), save_checkpoint)


if __name__ == "__main__":
    main()
