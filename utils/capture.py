from torch import Tensor
from torch.nn import Module
from transformers import PreTrainedModel


class _StopForward(Exception):
    pass


def capture_layer_inputs(
    model: PreTrainedModel,
    layer: Module,
    chunks: list[Tensor],
) -> list[tuple[tuple, dict]]:
    layer_inputs = []

    def hook(
        module: Module,
        args: tuple,
        kwargs: dict,
    ) -> None:
        layer_inputs.append((args, kwargs))
        raise _StopForward

    handle = layer.register_forward_pre_hook(hook, with_kwargs=True)
    for chunk in chunks:
        try:
            model(chunk[None, :].to(model.device), use_cache=False)
        except _StopForward:
            pass
    handle.remove()
    return layer_inputs
