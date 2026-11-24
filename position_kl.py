from dataclasses import dataclass

from simple_parsing import parse
import torch
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from utils.dataset import LANGUAGES, load_mgsm
from utils.hub import download_model, has_tensors, load_bf16_results, save_tensors
from utils.log import log_time_and_memory
from utils.model import load_model
from utils.path import extract_model_name
from utils.score import find_solved_problem_indices
from utils.stats import compute_reasoning_kls
from utils.tokenize import tokenize_prompt, tokenize_reasoning

RESULT_DIR = "position_kl/mgsm"


@dataclass
class Config:
    model_hub_path: str = "mxfp4/google/gemma-4-E2B-it"
    logit_batch_len: int = 2048
    max_generation_len: int = 16384
    reasoning_seed: int = 0
    bf16_seeds: tuple[int, ...] = (0, 1, 2)


def _compute_problem_kls(
    bf16_model: PreTrainedModel,
    quantized_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    prompts: list[str],
    bf16_reasonings: list[str],
    solved_problem_indices: list[int],
    logit_batch_len: int,
) -> tuple[Tensor, Tensor]:
    kls = []
    token_problem_indices = []
    for problem_index in solved_problem_indices:
        prompt_ids = tokenize_prompt(tokenizer, prompts[problem_index])
        reasoning_ids = tokenize_reasoning(tokenizer, model_name, bf16_reasonings[problem_index])
        kls.append(compute_reasoning_kls(bf16_model, quantized_model, torch.cat([prompt_ids, reasoning_ids]), len(prompt_ids), logit_batch_len))
        token_problem_indices.append(torch.full_like(reasoning_ids, problem_index, dtype=torch.int32))
    return torch.cat(kls), torch.cat(token_problem_indices)


@torch.no_grad()
def main() -> None:
    config = parse(Config)
    model_name = extract_model_name(config.model_hub_path)

    if has_tensors(RESULT_DIR, config.model_hub_path, config.max_generation_len, config.reasoning_seed):
        print(f"{config.model_hub_path} already evaluated")
        return

    problems = load_mgsm()
    bf16_results = load_bf16_results("mgsm", model_name, config.max_generation_len, config.bf16_seeds)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bf16_model = load_model(model_name)
    quantized_model = load_model(download_model(config.model_hub_path))

    tensors = {}
    for language in LANGUAGES:
        solved_problem_indices = find_solved_problem_indices(bf16_results, language)
        with log_time_and_memory(f"{config.model_hub_path} {language} {len(solved_problem_indices)} problems"):
            kls, token_problem_indices = _compute_problem_kls(bf16_model, quantized_model, tokenizer, model_name, problems[language]["prompts"], bf16_results[config.reasoning_seed][language]["reasonings"], solved_problem_indices, config.logit_batch_len)
        tensors[f"{language}/kls"] = kls
        tensors[f"{language}/token_problem_indices"] = token_problem_indices
        print(f"{config.model_hub_path} {language} kl {kls.mean():.4f}")

    save_tensors(RESULT_DIR, config.model_hub_path, config.max_generation_len, config.reasoning_seed, tensors)


if __name__ == "__main__":
    main()
