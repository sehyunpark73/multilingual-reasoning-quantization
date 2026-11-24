from collections.abc import Sequence
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from safetensors.torch import save
from torch import Tensor
from transformers import PreTrainedModel

REPO_NAME = "sehyunpark73/multilingual-reasoning-quantization"
LOSS_RESULT_DIR = "loss/mgsm"


def make_hub_path(
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


def make_finetune_hub_path(
    model_name: str,
    loss: str,
    reasoning_source: str,
    train_part: str,
) -> str:
    return f"finetune/{model_name}/loss={loss},reasoning_source={reasoning_source},train_part={train_part}"


def _make_result_path(
    result_dir: str,
    hub_path: str,
    max_generation_len: int,
    seed: int,
) -> str:
    return f"{result_dir}/{hub_path}/max_generation_len={max_generation_len},seed={seed}.json"


def _make_tensors_path(
    result_dir: str,
    hub_path: str,
    max_generation_len: int,
    reasoning_seed: int,
) -> str:
    return f"{result_dir}/{hub_path}/max_generation_len={max_generation_len},reasoning_seed={reasoning_seed}.safetensors"


def model_exists(
    hub_path: str,
) -> bool:
    return HfApi().file_exists(REPO_NAME, f"{hub_path}/config.json")


def download_model(
    hub_path: str,
    model_name: str,
) -> str:
    if hub_path == make_hub_path("bf16", model_name):
        return model_name
    return str(Path(snapshot_download(REPO_NAME, allow_patterns=[f"{hub_path}/*.json", f"{hub_path}/model*.safetensors"])) / hub_path)


def save_model(
    hub_path: str,
    model: PreTrainedModel,
) -> None:
    with TemporaryDirectory() as directory:
        model.save_pretrained(directory)
        HfApi().upload_folder(folder_path=directory, path_in_repo=hub_path, repo_id=REPO_NAME, commit_message=f"save {hub_path}")
    print(f"saved {hub_path}")


def result_exists(
    result_dir: str,
    hub_path: str,
    max_generation_len: int,
    seed: int,
) -> bool:
    return HfApi().file_exists(REPO_NAME, _make_result_path(result_dir, hub_path, max_generation_len, seed))


def load_results(
    result_dir: str,
    hub_path: str,
    max_generation_len: int,
    seeds: Sequence[int],
) -> dict[int, dict]:
    results = {}
    for seed in seeds:
        if result_exists(result_dir, hub_path, max_generation_len, seed):
            results[seed] = json.loads(Path(hf_hub_download(REPO_NAME, _make_result_path(result_dir, hub_path, max_generation_len, seed))).read_text())
        else:
            results[seed] = {}
    return results


def load_bf16_results(
    model_name: str,
    max_generation_len: int,
    seeds: Sequence[int],
) -> dict[int, dict]:
    return load_results(LOSS_RESULT_DIR, make_hub_path("bf16", model_name), max_generation_len, seeds)


def save_result(
    result_dir: str,
    hub_path: str,
    max_generation_len: int,
    seed: int,
    result: dict,
) -> None:
    path = _make_result_path(result_dir, hub_path, max_generation_len, seed)
    HfApi().upload_file(path_or_fileobj=json.dumps(result, indent=2, ensure_ascii=False).encode(), path_in_repo=path, repo_id=REPO_NAME, commit_message=f"save {path}")
    print(f"saved {path}")


def tensors_exist(
    result_dir: str,
    hub_path: str,
    max_generation_len: int,
    reasoning_seed: int,
) -> bool:
    return HfApi().file_exists(REPO_NAME, _make_tensors_path(result_dir, hub_path, max_generation_len, reasoning_seed))


def save_tensors(
    result_dir: str,
    hub_path: str,
    max_generation_len: int,
    reasoning_seed: int,
    tensors: dict[str, Tensor],
) -> None:
    path = _make_tensors_path(result_dir, hub_path, max_generation_len, reasoning_seed)
    HfApi().upload_file(path_or_fileobj=save(tensors), path_in_repo=path, repo_id=REPO_NAME, commit_message=f"save {path}")
    print(f"saved {path}")
