from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
import random
from typing import Literal, TypeAlias

from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn import Linear
from torch.nn.functional import log_softmax
from torch.nn.utils import parametrize
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, set_seed
from vllm import LLM  # type: ignore

from utils.dataset import NON_EN_LANGUAGES, load_translated_gsm8k
from utils.generate import compute_prompt_logprobs, generate_on_policy, load_on_policy_vllm
from utils.hub import has_model, load_result
from utils.log import log_time
from utils.model import MODEL_CONFIGS, compute_logits, find_linears, forward_reasoning_hiddens, load_model
from utils.path import make_model_hub_path
from utils.quantize import Fp4Method, QuantizeSte, quantize_fp4
from utils.tokenize import tokenize_en_question, tokenize_prompt
from utils.train import MAX_REASONING_LEN, Sample, compute_ce_kl_loss, compute_ce_loss, compute_kl_loss, make_en_question_samples, make_native_reasoning_samples, save_qat_checkpoint, train

FINETUNE_NAME = "qat_comparison"
COPSD_TEACHER_PROMPT = """{prompt}

English translation of the problem: {en_question}

Correct English reference solution:
=== Reference Solution Begin ===
{en_solution}
=== Reference Solution End ===

Read the English reference solution above until you understand why each step works, but do not copy or paraphrase it. Solve the original problem above with your own step-by-step reasoning and follow its answer format."""

_Loss: TypeAlias = Literal["ce_kl", "ce_opd", "pam", "qad", "copsd", "opd"]


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    method: Fp4Method = "nvfp4"
    loss: _Loss = "ce_kl"
    gsm8k_en_question_insert_reasoning_hub_path: str = "gsm8k_en_question_insert_reasoning/bf16/google/gemma-4-E2B-it/max_generation_len=16384,seed=0.json"
    gsm8k_reasoning_hub_path: str = "gsm8k_reasoning/bf16/google/gemma-4-E2B-it/max_generation_len=16384,seed=0.json"
    num_samples: int = 20000
    checkpoint_num_trained_samples: tuple[int, ...] = (5000, 10000, 15000, 20000)
    learning_rate: float = 1e-5
    num_epochs: int = 1
    warmup_ratio: float = 0.05
    batch_size: int = 8
    max_token_kl: float = 0.05
    num_batches_per_generation: int = 8
    train_seed: int = 0


def _make_on_policy_samples(
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    problems: dict[str, dict[str, list]],
    loss: _Loss,
) -> list[Sample]:
    empty_reasoning_ids = torch.tensor([], dtype=torch.long)
    samples = []
    for language in NON_EN_LANGUAGES:
        for prompt, en_question, en_solution in zip(problems[language]["prompts"], problems["en"]["questions"], problems["en"]["solutions"]):
            if loss == "ce_opd":
                samples.append((tokenize_prompt(tokenizer, prompt), tokenize_en_question(tokenizer, model_name, en_question), empty_reasoning_ids))
            if loss == "copsd":
                teacher_prompt = COPSD_TEACHER_PROMPT.format(prompt=prompt, en_question=en_question, en_solution=en_solution)
                samples.append((tokenize_prompt(tokenizer, prompt), tokenize_prompt(tokenizer, teacher_prompt), empty_reasoning_ids))
            if loss == "opd":
                samples.append((tokenize_prompt(tokenizer, prompt), tokenize_prompt(tokenizer, prompt), empty_reasoning_ids))
    return samples


def _make_generation_input_ids(
    loss: _Loss,
    sample: Sample,
) -> Tensor:
    input_ids = sample[0]
    if loss == "ce_opd":
        prompt_ids, en_question_ids, _ = sample
        input_ids = torch.cat([prompt_ids, en_question_ids])
    return input_ids


@torch.no_grad()
def _make_quantized_weights(
    model: PreTrainedModel,
    method: Fp4Method,
) -> list[tuple[str, Tensor]]:
    quantized_weights = []
    for name, module in model.model.language_model.layers.named_modules(prefix="model.language_model.layers"):
        if isinstance(module, Linear):
            quantized_weights.append((f"{name}.weight", quantize_fp4(module.parametrizations.weight.original, method).to(torch.bfloat16)))
    return quantized_weights


def _generate_on_policy_batches(
    vllm_model: LLM,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    method: Fp4Method,
    loss: _Loss,
    batches: list[list[Sample]],
) -> list[list[Sample]]:
    input_ids_list = []
    for batch in batches:
        for sample in batch:
            input_ids_list.append(_make_generation_input_ids(loss, sample))
    generation = generate_on_policy(vllm_model, model_name, _make_quantized_weights(model, method), input_ids_list, MAX_REASONING_LEN)
    reasoning_end_id = tokenizer.convert_tokens_to_ids(MODEL_CONFIGS[model_name]["reasoning_end"])
    on_policy_batches = []
    start = 0
    for batch in batches:
        on_policy_batch = []
        for sample, reasoning_ids, response in zip(batch, generation["reasoning_ids"][start : start + len(batch)], generation["responses"][start : start + len(batch)]):
            if response is not None:
                reasoning_ids = reasoning_ids + [reasoning_end_id]
            on_policy_batch.append(sample[:2] + (torch.tensor(reasoning_ids),))
        on_policy_batches.append(on_policy_batch)
        start += len(batch)
    return on_policy_batches


@torch.no_grad()
def _compute_vllm_logprob_errors(
    vllm_model: LLM,
    model: PreTrainedModel,
    bf16_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    method: Fp4Method,
    loss: _Loss,
    sample: Sample,
) -> tuple[float, float]:
    on_policy_sample = _generate_on_policy_batches(vllm_model, model, tokenizer, model_name, method, loss, [[sample]])[0][0]
    input_ids = torch.cat([_make_generation_input_ids(loss, on_policy_sample), on_policy_sample[2]])
    vllm_logprobs = compute_prompt_logprobs(vllm_model, input_ids)
    errors = []
    for hf_model in (model, bf16_model):
        with parametrize.cached(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = compute_logits(hf_model, forward_reasoning_hiddens(hf_model, input_ids, 1))
        logprobs = log_softmax(logits, dim=-1).gather(-1, input_ids[1:, None].to(logits.device))[:, 0]
        errors.append((logprobs.cpu() - vllm_logprobs).abs().mean().item())
    return errors[0], errors[1]


def _compute_losses(
    model: PreTrainedModel,
    bf16_model: PreTrainedModel | None,
    loss: _Loss,
    max_token_kl: float,
    batch: list[Sample],
) -> Iterator[Tensor]:
    if loss == "ce_kl":
        num_en_question_tokens = sum(len(en_question_ids) for _, en_question_ids, _ in batch)
        num_reasoning_tokens = sum(len(reasoning_ids) for _, _, reasoning_ids in batch)
        for prompt_ids, en_question_ids, reasoning_ids in batch:
            yield compute_ce_kl_loss(model, bf16_model, prompt_ids, en_question_ids, reasoning_ids, num_en_question_tokens, num_reasoning_tokens)
    if loss == "ce_opd":
        num_en_question_tokens = sum(len(en_question_ids) for _, en_question_ids, _ in batch)
        num_reasoning_tokens = sum(len(reasoning_ids) for _, _, reasoning_ids in batch)
        for prompt_ids, en_question_ids, reasoning_ids in batch:
            yield compute_ce_kl_loss(model, bf16_model, prompt_ids, en_question_ids, reasoning_ids, num_en_question_tokens, num_reasoning_tokens, max_token_kl=max_token_kl)
    if loss == "pam":
        num_text_tokens = sum(len(en_question_ids) + len(reasoning_ids) for _, en_question_ids, reasoning_ids in batch)
        for prompt_ids, en_question_ids, reasoning_ids in batch:
            yield compute_ce_loss(model, prompt_ids, torch.cat([en_question_ids, reasoning_ids])) / num_text_tokens
    if loss == "qad":
        num_reasoning_tokens = sum(len(reasoning_ids) for _, _, reasoning_ids in batch)
        for prompt_ids, _, reasoning_ids in batch:
            yield compute_kl_loss(model, bf16_model, prompt_ids, prompt_ids, reasoning_ids) / num_reasoning_tokens
    if loss in ("copsd", "opd"):
        num_reasoning_tokens = sum(len(reasoning_ids) for _, _, reasoning_ids in batch)
        for prompt_ids, teacher_prompt_ids, reasoning_ids in batch:
            yield compute_kl_loss(model, bf16_model, prompt_ids, teacher_prompt_ids, reasoning_ids, max_token_kl=max_token_kl) / num_reasoning_tokens


def main() -> None:
    config = parse(Config)
    num_trained_samples = config.num_epochs * config.num_samples
    model_hub_path = make_model_hub_path(FINETUNE_NAME, config.model_name, {"method": config.method, "loss": config.loss, "num_samples": config.num_samples, "num_trained_samples": num_trained_samples})

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
    bf16_model = None
    if config.loss in ("ce_kl", "ce_opd", "qad", "copsd", "opd"):
        bf16_model = load_model(config.model_name)
    vllm_model = None
    if config.loss in ("ce_opd", "copsd", "opd"):
        vllm_model = load_on_policy_vllm(config.model_name, config.train_seed)

    if config.loss in ("ce_kl", "pam"):
        gsm8k_en_question_insert_reasoning_result = load_result(config.gsm8k_en_question_insert_reasoning_hub_path)
        all_samples = make_en_question_samples(tokenizer, config.model_name, gsm8k_en_question_insert_reasoning_result)
    if config.loss == "qad":
        gsm8k_reasoning_result = load_result(config.gsm8k_reasoning_hub_path)
        all_samples = make_native_reasoning_samples(tokenizer, config.model_name, gsm8k_reasoning_result)
    if config.loss in ("ce_opd", "copsd", "opd"):
        problems = load_translated_gsm8k()
        all_samples = _make_on_policy_samples(tokenizer, config.model_name, problems, config.loss)
    samples = random.sample(all_samples, config.num_samples)
    print(f"{model_hub_path} samples {len(samples)} of {len(all_samples)}")

    with log_time(f"{model_hub_path} finetune"):
        linears = find_linears(model.model.language_model.layers)
        for linear in linears:
            parametrize.register_parametrization(linear, "weight", QuantizeSte(config.method))
        weights = [linear.parametrizations.weight.original for linear in linears]
        if vllm_model is not None:
            quantized_error, bf16_error = _compute_vllm_logprob_errors(vllm_model, model, bf16_model, tokenizer, config.model_name, config.method, config.loss, samples[0])
            print(f"{model_hub_path} vllm logprob error quantized {quantized_error:.4f} bf16 {bf16_error:.4f}")
            if quantized_error >= bf16_error:
                raise ValueError(f"vllm weights are not updated, logprob error quantized {quantized_error:.4f} bf16 {bf16_error:.4f}")
        save_checkpoint = partial(save_qat_checkpoint, model, linears, weights, config.method, FINETUNE_NAME, config.model_name, {"method": config.method, "loss": config.loss, "num_samples": config.num_samples})
        generate_batches = None
        if vllm_model is not None:
            generate_batches = partial(_generate_on_policy_batches, vllm_model, model, tokenizer, config.model_name, config.method, config.loss)
        train(weights, samples, partial(_compute_losses, model, bf16_model, config.loss, config.max_token_kl), config.learning_rate, config.num_epochs, config.warmup_ratio, config.batch_size, config.checkpoint_num_trained_samples + (num_trained_samples,), save_checkpoint, generate_batches, config.num_batches_per_generation)


if __name__ == "__main__":
    main()
