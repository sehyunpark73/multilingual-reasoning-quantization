from dataclasses import dataclass

from simple_parsing import parse
import torch
from torch import Tensor

from utils.hub import save_model
from utils.model import find_linears, load_model
from utils.path import make_model_hub_path

NF4_GROUP_SIZE = 64
MIN_SCALE = 1e-8
NF4_VALUES = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"


def _quantize_weight(
    weight: Tensor,
) -> Tensor:
    groups = weight.float().reshape(weight.shape[0], -1, NF4_GROUP_SIZE)
    scale = groups.abs().amax(dim=-1, keepdim=True).clamp(min=MIN_SCALE)
    values = torch.tensor(NF4_VALUES, device=weight.device)
    value_indices = torch.bucketize(groups / scale, (values[1:] + values[:-1]) / 2)
    return (values[value_indices] * scale).reshape(weight.shape)


@torch.no_grad()
def main() -> None:
    config = parse(Config)
    model = load_model(config.model_name)
    for linear in find_linears(model.model.language_model.layers):
        linear.weight.copy_(_quantize_weight(linear.weight))
    save_model(make_model_hub_path("nf4", config.model_name), model)


if __name__ == "__main__":
    main()
