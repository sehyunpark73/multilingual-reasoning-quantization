from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from typing import Literal

from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn.functional import cosine_similarity, cross_entropy
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, set_seed

from utils.dataset import LANGUAGES
from utils.hub import has_model, load_result, save_model
from utils.log import log_time_and_memory
from utils.model import MODEL_CONFIGS, compute_logits, find_linears, forward_reasoning_hiddens, load_model
from utils.stats import compute_kls
from utils.tokenize import tokenize_prompt, tokenize_reasoning
from utils.train import train
from utils.understanding import extract_understanding

TRAIN_LANGUAGES = [language for language in LANGUAGES if language != "en"]
ON_POLICY_MAX_GENERATION_LENS = {"understanding": 512, "reasoning": 2048}

_Sample = tuple[Tensor, Tensor, Tensor]


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    loss: Literal["ce", "kl", "hidden"] = "ce"
    reasoning_source: Literal["bf16_en", "on_policy"] = "bf16_en"
    train_part: Literal["understanding", "reasoning"] = "understanding"
    gsm8k_reasoning_hub_path: str = "gsm8k_reasoning/bf16/google/gemma-4-E2B-it/max_generation_len=16384,seed=0.json"
    num_samples: int = 4000
    learning_rate: float = 1e-5
    num_epochs: int = 1
    warmup_ratio: float = 0.05
    batch_size: int = 8
    train_seed: int = 0


def _make_samples(
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    bf16_result: dict,
    train_part: str,
) -> list[_Sample]:
    samples = []
    for problem_index, en_reasoning in enumerate(bf16_result["en"]["reasonings"]):
        en_understanding = extract_understanding(en_reasoning)
        if en_understanding is None:
            continue
        bf16_prompt_ids = tokenize_prompt(tokenizer, bf16_result["en"]["prompts"][problem_index])
        text_ids = tokenize_reasoning(tokenizer, model_name, en_understanding if train_part == "understanding" else en_reasoning)
        for language in TRAIN_LANGUAGES:
            if extract_understanding(bf16_result[language]["reasonings"][problem_index]) is None or not bf16_result[language]["is_corrects"][problem_index]:
                continue
            samples.append((tokenize_prompt(tokenizer, bf16_result[language]["prompts"][problem_index]), bf16_prompt_ids, text_ids))
    return samples


def _generate_on_policy_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    train_part: str,
    batch: list[_Sample],
) -> list[_Sample]:
    sampling = MODEL_CONFIGS[model_name]["sampling"]
    inputs = tokenizer.pad({"input_ids": [prompt_ids for prompt_ids, _, _ in batch]}, padding_side="left", return_tensors="pt").to(model.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_ids = model.generate(**inputs, do_sample=True, temperature=sampling["temperature"], top_p=sampling["top_p"], top_k=sampling["top_k"], max_new_tokens=ON_POLICY_MAX_GENERATION_LENS[train_part])

    reasoning_end_id = tokenizer.convert_tokens_to_ids(MODEL_CONFIGS[model_name]["reasoning_end"])
    on_policy_batch = []
    for (prompt_ids, bf16_prompt_ids, _), generated_ids in zip(batch, output_ids[:, inputs["input_ids"].shape[1] :]):
        reasoning_end_positions = (generated_ids == reasoning_end_id).nonzero()
        reasoning = tokenizer.decode(generated_ids[: reasoning_end_positions[0, 0]] if len(reasoning_end_positions) else generated_ids, skip_special_tokens=True)
        understanding = extract_understanding(reasoning)
        if understanding is not None:
            on_policy_batch.append((prompt_ids, bf16_prompt_ids, tokenize_reasoning(tokenizer, model_name, understanding if train_part == "understanding" else reasoning)))
    return on_policy_batch


def _forward_layer_hiddens(
    model: PreTrainedModel,
    input_ids: Tensor,
    prompt_len: int,
) -> Tensor:
    hidden_states = model.model(input_ids=input_ids[None, :].to(model.device), output_hidden_states=True, use_cache=False).hidden_states
    return torch.stack(hidden_states[1:])[:, 0, prompt_len:]


def _compute_sample_loss(
    model: PreTrainedModel,
    bf16_model: PreTrainedModel | None,
    loss: str,
    prompt_ids: Tensor,
    bf16_prompt_ids: Tensor,
    text_ids: Tensor,
) -> Tensor:
    input_ids = torch.cat([prompt_ids, text_ids])
    bf16_input_ids = torch.cat([bf16_prompt_ids, text_ids])
    if loss == "ce":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = compute_logits(model, forward_reasoning_hiddens(model, input_ids, len(prompt_ids)))
        return cross_entropy(logits, text_ids.to(logits.device), reduction="sum")
    if loss == "kl":
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            bf16_logits = compute_logits(bf16_model, forward_reasoning_hiddens(bf16_model, bf16_input_ids, len(bf16_prompt_ids)))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = compute_logits(model, forward_reasoning_hiddens(model, input_ids, len(prompt_ids)))
        return compute_kls(bf16_logits, logits).sum()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        bf16_hiddens = _forward_layer_hiddens(bf16_model, bf16_input_ids, len(bf16_prompt_ids))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hiddens = _forward_layer_hiddens(model, input_ids, len(prompt_ids))
    return (1 - cosine_similarity(hiddens.float(), bf16_hiddens.float(), dim=-1)).mean(0).sum()


def _compute_losses(
    model: PreTrainedModel,
    bf16_model: PreTrainedModel | None,
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    loss: str,
    reasoning_source: str,
    train_part: str,
    batch: list[_Sample],
) -> Iterator[Tensor]:
    if reasoning_source == "on_policy":
        batch = _generate_on_policy_batch(model, tokenizer, model_name, train_part, batch)
    num_tokens = sum(len(text_ids) for _, _, text_ids in batch)
    for prompt_ids, bf16_prompt_ids, text_ids in batch:
        yield _compute_sample_loss(model, bf16_model, loss, prompt_ids, bf16_prompt_ids, text_ids) / num_tokens


def main() -> None:
    config = parse(Config)
    model_hub_path = f"finetune/{config.model_name}/loss={config.loss},reasoning_source={config.reasoning_source},train_part={config.train_part}"

    if has_model(model_hub_path):
        print(f"{model_hub_path} already finetuned")
        return

    set_seed(config.train_seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = load_model(config.model_name)
    model.float()
    bf16_model = None if config.loss == "ce" else load_model(config.model_name)

    bf16_result = load_result(config.gsm8k_reasoning_hub_path)
    samples = _make_samples(tokenizer, config.model_name, bf16_result, config.train_part)[: config.num_samples]
    print(f"{model_hub_path} samples {len(samples)}")

    with log_time_and_memory(f"{model_hub_path} finetune"):
        weights = [linear.weight for linear in find_linears(model.model.language_model.layers)]
        train(weights, samples, partial(_compute_losses, model, bf16_model, tokenizer, config.model_name, config.loss, config.reasoning_source, config.train_part), config.learning_rate, config.num_epochs, config.warmup_ratio, config.batch_size)

    model.to(torch.bfloat16)
    save_model(model_hub_path, model)


if __name__ == "__main__":
    main()
