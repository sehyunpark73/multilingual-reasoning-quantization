import torch
from torch import Tensor

MXFP4_GROUP_SIZE = 32
MIN_WEIGHT_MAX = 2**-125
FP4_MAX_EXPONENT = 2
FP4_MAX_MAGNITUDE = 6


def quantize_mxfp4(
    weight: Tensor,
) -> Tensor:
    groups = weight.float().reshape(weight.shape[0], -1, MXFP4_GROUP_SIZE)
    weight_max = groups.abs().amax(dim=-1, keepdim=True).clamp(min=MIN_WEIGHT_MAX)
    scale = 2 ** (torch.floor(torch.log2(weight_max)) - FP4_MAX_EXPONENT)
    magnitude = (groups.abs() / scale).clamp(max=FP4_MAX_MAGNITUDE)
    step = torch.where(magnitude < 2, 0.5, torch.where(magnitude < 4, 1.0, 2.0))
    return (torch.sign(groups) * torch.round(magnitude / step) * step * scale).reshape(weight.shape)
