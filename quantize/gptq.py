from dataclasses import dataclass
from typing import Literal

from simple_parsing import parse
import torch
from torch import Tensor

from utils.dataset import load_wikipedia
from utils.grid import compute_grid, round_to_grid
from utils.hub import save_model
from utils.log import log_time_and_memory
from utils.model import capture_layer_inputs, find_linears, load_model
from utils.path import make_model_hub_path
from utils.stats import compute_abs_mean_and_hessian


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    calibration_languages: tuple[Literal["en", "zh", "ja", "sw", "te"], ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_chunks: int = 128
    chunk_len: int = 2048
    shuffle_seed: int = 0
    block_size: int = 128
    damp_ratio: float = 0.01


def _quantize_weight(
    weight: Tensor,
    hessian: Tensor,
    num_bits: int,
    group_size: int,
    block_size: int,
    damp_ratio: float,
) -> Tensor:
    weight = weight.float().clone()
    is_zero_diagonal = hessian.diagonal() == 0
    weight[:, is_zero_diagonal] = 0
    hessian.diagonal()[is_zero_diagonal] = 1
    hessian.diagonal().add_(damp_ratio * hessian.diagonal().mean())
    hessian_inverse = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(hessian)), upper=True)

    quantized_weight = torch.zeros_like(weight)
    scale = torch.zeros(weight.shape[0], weight.shape[1] // group_size, device=weight.device)
    zero = torch.zeros(weight.shape[0], weight.shape[1] // group_size, device=weight.device)
    error = torch.zeros(weight.shape[0], block_size, device=weight.device)
    for block_start in range(0, weight.shape[1], block_size):
        block_end = min(block_start + block_size, weight.shape[1])
        for column_index in range(block_start, block_end):
            group_index = column_index // group_size
            if column_index % group_size == 0:
                scale[:, group_index], zero[:, group_index] = compute_grid(weight[:, column_index : column_index + group_size], num_bits)
            quantized_weight[:, column_index] = round_to_grid(weight[:, column_index], scale[:, group_index], zero[:, group_index], num_bits)
            error[:, column_index - block_start] = (weight[:, column_index] - quantized_weight[:, column_index]) / hessian_inverse[column_index, column_index]
            weight[:, column_index:block_end] -= torch.outer(error[:, column_index - block_start], hessian_inverse[column_index, column_index:block_end])
        weight[:, block_end:] -= error[:, : block_end - block_start] @ hessian_inverse[block_start:block_end, block_end:]
    return quantized_weight


@torch.no_grad()
def main() -> None:
    config = parse(Config)
    model = load_model(config.model_name)
    chunks = load_wikipedia(config.model_name, config.calibration_languages, config.num_chunks, config.chunk_len, config.shuffle_seed)
    layers = model.model.language_model.layers

    for layer_index, layer in enumerate(layers):
        with log_time_and_memory(f"layer {layer_index + 1}/{len(layers)}"):
            layer_inputs = capture_layer_inputs(model, layer, chunks)
            for linear in find_linears(layer):
                _, hessian = compute_abs_mean_and_hessian(layer, linear, layer_inputs)
                linear.weight.copy_(_quantize_weight(linear.weight, hessian, config.num_bits, config.group_size, config.block_size, config.damp_ratio))

    save_model(make_model_hub_path("gptq", config.model_name, config.calibration_languages, config.num_bits, config.group_size, config.num_chunks), model)


if __name__ == "__main__":
    main()
