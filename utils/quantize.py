from typing import Literal, TypeAlias

import torch
from torch import Tensor
from torch.nn import Module

MXFP4_GROUP_SIZE = 32
NVFP4_GROUP_SIZE = 16
MIN_WEIGHT_MAX = 2**-125
FP4_MAX_EXPONENT = 2
FP4_MAX_MAGNITUDE = 6
FP8_MAX_MAGNITUDE = 448

Fp4Method: TypeAlias = Literal["mxfp4", "nvfp4"]


def _round_to_fp4(
    groups: Tensor,
    scale: Tensor,
) -> Tensor:
    magnitude = (groups.abs() / scale).clamp(max=FP4_MAX_MAGNITUDE)
    step = torch.where(magnitude < 2, 0.5, torch.where(magnitude < 4, 1.0, 2.0))
    return torch.sign(groups) * torch.round(magnitude / step) * step * scale


def quantize_mxfp4(
    weight: Tensor,
) -> Tensor:
    groups = weight.float().reshape(weight.shape[0], -1, MXFP4_GROUP_SIZE)
    weight_max = groups.abs().amax(dim=-1, keepdim=True).clamp(min=MIN_WEIGHT_MAX)
    scale = 2 ** (torch.floor(torch.log2(weight_max)) - FP4_MAX_EXPONENT)
    return _round_to_fp4(groups, scale).reshape(weight.shape)


def quantize_nvfp4(
    weight: Tensor,
) -> Tensor:
    groups = weight.float().reshape(weight.shape[0], -1, NVFP4_GROUP_SIZE)
    tensor_scale = groups.abs().amax().clamp(min=MIN_WEIGHT_MAX) / (FP4_MAX_MAGNITUDE * FP8_MAX_MAGNITUDE)
    group_scale = (groups.abs().amax(dim=-1, keepdim=True) / FP4_MAX_MAGNITUDE / tensor_scale).to(torch.float8_e4m3fn).float()
    scale = (group_scale * tensor_scale).clamp(min=MIN_WEIGHT_MAX)
    return _round_to_fp4(groups, scale).reshape(weight.shape)


def quantize_fp4(
    weight: Tensor,
    method: Fp4Method,
) -> Tensor:
    if method == "mxfp4":
        quantized_weight = quantize_mxfp4(weight)
    if method == "nvfp4":
        quantized_weight = quantize_nvfp4(weight)
    return quantized_weight


class QuantizeSte(Module):
    def __init__(
        self,
        method: Fp4Method,
    ) -> None:
        super().__init__()
        self.method = method

    def forward(
        self,
        weight: Tensor,
    ) -> Tensor:
        with torch.no_grad():
            quantized_weight = quantize_fp4(weight, self.method)
        return weight + (quantized_weight - weight).detach()
