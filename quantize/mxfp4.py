from dataclasses import dataclass

from simple_parsing import parse
import torch

from utils.hub import download_model, save_model
from utils.model import find_linears, load_model
from utils.path import make_model_hub_path, make_quantized_finetune_model_hub_path
from utils.quantize import quantize_mxfp4


@dataclass
class Config:
    model_name: str = "google/gemma-4-E2B-it"
    finetune_model_hub_path: str | None = None


@torch.no_grad()
def main() -> None:
    config = parse(Config)
    if config.finetune_model_hub_path is None:
        model = load_model(config.model_name)
        model_hub_path = make_model_hub_path("mxfp4", config.model_name)
    else:
        model = load_model(download_model(config.finetune_model_hub_path))
        model_hub_path = make_quantized_finetune_model_hub_path(config.finetune_model_hub_path, "mxfp4")
    for linear in find_linears(model.model.language_model.layers):
        linear.weight.copy_(quantize_mxfp4(linear.weight))
    save_model(model_hub_path, model)


if __name__ == "__main__":
    main()
