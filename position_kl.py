from dataclasses import dataclass
from typing import Literal

from simple_parsing import parse
import torch
from transformers import AutoTokenizer

from utils.dataset import LANGUAGES, load_mgsm
from utils.hub import download_model, load_results, make_hub_path, save_tensors, tensors_exist
from utils.kl import compute_reasoning_kls
from utils.log import log_time_and_memory
from utils.model import load_model
from utils.score import find_solved_problem_indices
from utils.tokenize import tokenize_prompt, tokenize_reasoning

RESULT_DIR = "position_kl/mgsm"


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    method: Literal["gptq", "awq", "nf4", "mxfp4"] = "mxfp4"
    calibration_languages: tuple[Literal["en", "zh", "ja", "sw", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_chunks: int = 128
    logit_batch_len: int = 2048
    max_generation_len: int = 16384
    reasoning_seed: int = 0
    bf16_seeds: tuple[int, ...] = (0, 1, 2)


@torch.no_grad()
def main() -> None:
    config = parse(Config)
    hub_path = make_hub_path(config.method, config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_chunks)
    if tensors_exist(RESULT_DIR, hub_path, config.max_generation_len, config.reasoning_seed):
        print(f"{hub_path} already evaluated")
        return

    problems = load_mgsm()
    bf16_hub_path = make_hub_path("bf16", config.model_name)
    bf16_results = load_results("loss/mgsm", bf16_hub_path, config.max_generation_len, config.bf16_seeds)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    bf16_model = load_model(config.model_name)
    quantized_model = load_model(download_model(hub_path))

    tensors = {}
    for language in LANGUAGES:
        problem_indices = find_solved_problem_indices(bf16_results, language)
        kls = []
        token_problem_indices = []
        with log_time_and_memory(f"{hub_path} {language} {len(problem_indices)} problems"):
            for problem_index in problem_indices:
                prompt_ids = tokenize_prompt(tokenizer, problems[language]["prompts"][problem_index])
                reasoning_ids = tokenize_reasoning(tokenizer, config.model_name, bf16_results[config.reasoning_seed][language]["reasonings"][problem_index])
                kls.append(compute_reasoning_kls(bf16_model, quantized_model, torch.cat([prompt_ids, reasoning_ids]), len(prompt_ids), config.logit_batch_len))
                token_problem_indices.append(torch.full_like(reasoning_ids, problem_index, dtype=torch.int32))
        tensors[f"{language}/kls"] = torch.cat(kls)
        tensors[f"{language}/token_problem_indices"] = torch.cat(token_problem_indices)
        print(f"{hub_path} {language} kl {tensors[f'{language}/kls'].mean():.4f}")

    save_tensors(RESULT_DIR, hub_path, config.max_generation_len, config.reasoning_seed, tensors)


if __name__ == "__main__":
    main()
