from utils.dataset import Benchmark


def make_model_hub_path(
    method: str,
    model_name: str,
    calibration_languages: tuple[str, ...] | None = None,
    num_bits: int | None = None,
    group_size: int | None = None,
    num_chunks: int | None = None,
) -> str:
    if method in ("bf16", "nf4", "mxfp4"):
        return f"{method}/{model_name}"
    return f"{method}/{model_name}/calibration_languages={'_'.join(calibration_languages)},num_bits={num_bits},group_size={group_size},num_chunks={num_chunks}"


def make_finetune_model_hub_path(
    finetune_name: str,
    model_name: str,
    settings: dict[str, object],
) -> str:
    return f"{finetune_name}/{model_name}/{','.join(f'{name}={value}' for name, value in settings.items())}"


def extract_model_name(
    model_hub_path: str,
) -> str:
    _, organization, name = model_hub_path.split("/")[:3]
    return f"{organization}/{name}"


def make_loss_result_dir(
    benchmark: Benchmark,
) -> str:
    return f"loss/{benchmark}"


def make_result_path(
    result_dir: str,
    model_hub_path: str,
    max_generation_len: int,
    seed: int,
) -> str:
    return f"{result_dir}/{model_hub_path}/max_generation_len={max_generation_len},seed={seed}.json"


def make_tensors_path(
    result_dir: str,
    model_hub_path: str,
    max_generation_len: int,
    reasoning_seed: int,
) -> str:
    return f"{result_dir}/{model_hub_path}/max_generation_len={max_generation_len},reasoning_seed={reasoning_seed}.safetensors"
