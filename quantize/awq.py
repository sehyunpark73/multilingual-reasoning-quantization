from dataclasses import dataclass
import math

import numpy as np
from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn import Linear, Module

from utils.dataset import Language, load_wikipedia
from utils.grid import compute_grid, round_to_grid
from utils.hub import save_model
from utils.log import log_time_and_memory
from utils.model import capture_layer_inputs, find_linears, load_model
from utils.path import make_model_hub_path
from utils.stats import compute_abs_mean_and_hessian

MIN_SMOOTH = 1e-4


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    calibration_languages: tuple[Language, ...] = ("en",)
    num_bits: int = 4
    group_size: int = 128
    num_chunks: int = 128
    chunk_len: int = 2048
    shuffle_seed: int = 0
    num_search_steps: int = 20
    max_shrink: float = 0.5


def _find_linear_groups(
    layer: Module,
    layer_args: tuple,
    layer_kwargs: dict,
) -> list[list[Linear]]:
    group_activations = []
    linear_groups = []

    def hook(
        module: Linear,
        args: tuple[Tensor, ...],
    ) -> None:
        for activations, linears in zip(group_activations, linear_groups):
            if activations is args[0]:
                linears.append(module)
                return
        group_activations.append(args[0])
        linear_groups.append([module])

    handles = [linear.register_forward_pre_hook(hook) for linear in find_linears(layer)]
    layer(*layer_args, **layer_kwargs)
    for handle in handles:
        handle.remove()
    return linear_groups


def _quantize_weight(
    weight: Tensor,
    smooth: Tensor,
    clip: Tensor | None,
    num_bits: int,
    group_size: int,
) -> Tensor:
    groups = (weight.float() * smooth).reshape(weight.shape[0], -1, group_size)
    clipped_groups = groups if clip is None else groups.clamp(-clip[..., None], clip[..., None])
    scale, zero = compute_grid(clipped_groups, num_bits)
    return round_to_grid(groups, scale[..., None], zero[..., None], num_bits).reshape(weight.shape) / smooth


def _find_smooth(
    layer: Module,
    linears: list[Linear],
    layer_inputs: list[tuple[tuple, dict]],
    abs_mean: Tensor,
    num_bits: int,
    group_size: int,
    num_search_steps: int,
) -> Tensor:
    unquantized_outputs = [layer(*layer_args, **layer_kwargs) for layer_args, layer_kwargs in layer_inputs]
    unquantized_weights = [linear.weight.clone() for linear in linears]

    best_error = math.inf
    best_smooth = torch.ones_like(abs_mean)
    for smooth_exponent in np.linspace(0.0, 1.0, num_search_steps, endpoint=False):
        smooth = abs_mean.pow(smooth_exponent).clamp(min=MIN_SMOOTH)
        smooth = smooth / (smooth.max() * smooth.min()).sqrt()
        for linear, unquantized_weight in zip(linears, unquantized_weights):
            linear.weight.copy_(_quantize_weight(unquantized_weight, smooth, None, num_bits, group_size))
        error = 0.0
        for (layer_args, layer_kwargs), unquantized_output in zip(layer_inputs, unquantized_outputs):
            quantized_output = layer(*layer_args, **layer_kwargs)
            error += (unquantized_output - quantized_output).float().pow(2).mean().item()
        if error < best_error:
            best_error = error
            best_smooth = smooth
    for linear, unquantized_weight in zip(linears, unquantized_weights):
        linear.weight.copy_(unquantized_weight)
    return best_smooth


def _find_clip(
    weight: Tensor,
    smooth: Tensor,
    hessian: Tensor,
    num_bits: int,
    group_size: int,
    num_search_steps: int,
    max_shrink: float,
) -> Tensor:
    groups = (weight.float() * smooth).reshape(weight.shape[0], -1, group_size)
    num_groups = groups.shape[1]
    smoothed_hessian = hessian / smooth[:, None] / smooth[None, :]
    hessian_blocks = smoothed_hessian.reshape(num_groups, group_size, num_groups, group_size).diagonal(dim1=0, dim2=2).permute(2, 0, 1)

    weight_max = groups.abs().amax(dim=-1)
    best_clip = weight_max.clone()
    best_error = torch.full_like(weight_max, math.inf)
    for shrink in np.linspace(1.0, 1.0 - max_shrink, int(max_shrink * num_search_steps), endpoint=False):
        clip = weight_max * shrink
        scale, zero = compute_grid(groups.clamp(-clip[..., None], clip[..., None]), num_bits)
        quantized_groups = round_to_grid(groups, scale[..., None], zero[..., None], num_bits)
        delta = quantized_groups - groups
        error = torch.einsum("ogi,gij,ogj->og", delta, hessian_blocks, delta)
        is_better = error < best_error
        best_error = torch.where(is_better, error, best_error)
        best_clip = torch.where(is_better, clip, best_clip)
    return best_clip


@torch.no_grad()
def main() -> None:
    config = parse(Config)
    model = load_model(config.model_name)
    chunks = load_wikipedia(config.model_name, config.calibration_languages, config.num_chunks, config.chunk_len, config.shuffle_seed)
    layers = model.model.language_model.layers

    for layer_index, layer in enumerate(layers):
        with log_time_and_memory(f"layer {layer_index + 1}/{len(layers)}"):
            layer_inputs = capture_layer_inputs(model, layer, chunks)
            layer_args, layer_kwargs = layer_inputs[0]
            for linears in _find_linear_groups(layer, layer_args, layer_kwargs):
                abs_mean, hessian = compute_abs_mean_and_hessian(layer, linears[0], layer_inputs)
                smooth = _find_smooth(layer, linears, layer_inputs, abs_mean, config.num_bits, config.group_size, config.num_search_steps)
                for linear in linears:
                    clip = _find_clip(linear.weight, smooth, hessian, config.num_bits, config.group_size, config.num_search_steps, config.max_shrink)
                    linear.weight.copy_(_quantize_weight(linear.weight, smooth, clip, config.num_bits, config.group_size))

    settings = {"calibration_languages": "_".join(config.calibration_languages), "num_bits": config.num_bits, "group_size": config.group_size, "num_chunks": config.num_chunks}
    save_model(make_model_hub_path("awq", config.model_name, settings), model)


if __name__ == "__main__":
    main()
