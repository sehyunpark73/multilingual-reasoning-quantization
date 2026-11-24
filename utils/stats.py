import torch
from torch import Tensor
from torch.nn import Linear, Module


def compute_abs_mean_and_hessian(
    layer: Module,
    linear: Linear,
    layer_inputs: list[tuple[tuple, dict]],
) -> tuple[Tensor, Tensor]:
    abs_sum = torch.zeros(linear.in_features, device=linear.weight.device)
    hessian = torch.zeros(linear.in_features, linear.in_features, device=linear.weight.device)
    num_tokens = 0

    def hook(
        module: Module,
        args: tuple[Tensor, ...],
    ) -> None:
        nonlocal num_tokens
        activations = args[0].reshape(-1, args[0].shape[-1]).float()
        abs_sum.add_(activations.abs().sum(dim=0))
        hessian.add_(activations.T @ activations)
        num_tokens += activations.shape[0]

    handle = linear.register_forward_pre_hook(hook)
    for layer_args, layer_kwargs in layer_inputs:
        layer(*layer_args, **layer_kwargs)
    handle.remove()
    return abs_sum / num_tokens, hessian
