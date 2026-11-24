from dataclasses import dataclass

from simple_parsing import parse
import torch
from torch import Tensor

from utils.hub import make_hub_path, save_model
from utils.model import find_linears, load_model

MXFP4_GROUP_SIZE = 32
MIN_WEIGHT_MAX = 2**-125
FP4_MAX_EXPONENT = 2
FP4_MAX_MAGNITUDE = 6


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"


def _quantize_weight(
    weight: Tensor,
) -> Tensor:
    groups = weight.float().reshape(weight.shape[0], -1, MXFP4_GROUP_SIZE)
    weight_max = groups.abs().amax(dim=-1, keepdim=True).clamp(min=MIN_WEIGHT_MAX)
    scale = 2 ** (torch.floor(torch.log2(weight_max)) - FP4_MAX_EXPONENT)
    magnitude = (groups.abs() / scale).clamp(max=FP4_MAX_MAGNITUDE)
    step = torch.where(magnitude < 2, 0.5, torch.where(magnitude < 4, 1.0, 2.0))
    return (torch.sign(groups) * torch.round(magnitude / step) * step * scale).reshape(weight.shape)


@torch.no_grad()
def main() -> None:
    config = parse(Config)
    model = load_model(config.model_name)
    for linear in find_linears(model.model.language_model.layers):
        linear.weight.copy_(_quantize_weight(linear.weight))
    save_model(make_hub_path("mxfp4", config.model_name), model)


if __name__ == "__main__":
    main()
