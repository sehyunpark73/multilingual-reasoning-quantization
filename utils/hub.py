from collections.abc import Sequence
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from safetensors.torch import load_file, save
from torch import Tensor
from transformers import PreTrainedModel

from utils.dataset import Benchmark
from utils.path import extract_model_name, make_model_hub_path, make_result_dir

REPO_NAME = "sehyunpark73/multilingual-reasoning-quantization"


def _make_result_path(
    result_dir: str,
    model_hub_path: str,
    max_generation_len: int,
    seed: int,
) -> str:
    return f"{result_dir}/{model_hub_path}/max_generation_len={max_generation_len},seed={seed}.json"


def _make_tensors_path(
    result_dir: str,
    model_hub_path: str,
    max_generation_len: int,
    reasoning_seed: int,
) -> str:
    return f"{result_dir}/{model_hub_path}/max_generation_len={max_generation_len},reasoning_seed={reasoning_seed}.safetensors"


def has_model(
    model_hub_path: str,
) -> bool:
    return HfApi().file_exists(REPO_NAME, f"{model_hub_path}/config.json")


def download_model(
    model_hub_path: str,
) -> str:
    model_name = extract_model_name(model_hub_path)
    if model_hub_path == make_model_hub_path("bf16", model_name):
        return model_name
    return str(Path(snapshot_download(REPO_NAME, allow_patterns=[f"{model_hub_path}/*.json", f"{model_hub_path}/model*.safetensors"])) / model_hub_path)


def save_model(
    model_hub_path: str,
    model: PreTrainedModel,
) -> None:
    with TemporaryDirectory() as directory:
        model.save_pretrained(directory)
        HfApi().upload_folder(folder_path=directory, path_in_repo=model_hub_path, repo_id=REPO_NAME, commit_message=f"save {model_hub_path}")
    print(f"saved {model_hub_path}")


def has_result(
    result_dir: str,
    model_hub_path: str,
    max_generation_len: int,
    seed: int,
) -> bool:
    return HfApi().file_exists(REPO_NAME, _make_result_path(result_dir, model_hub_path, max_generation_len, seed))


def load_result(
    result_hub_path: str,
) -> dict:
    return json.loads(Path(hf_hub_download(REPO_NAME, result_hub_path)).read_text())


def load_results(
    result_dir: str,
    model_hub_path: str,
    max_generation_len: int,
    seeds: Sequence[int],
) -> dict[int, dict]:
    results = {}
    for seed in seeds:
        if has_result(result_dir, model_hub_path, max_generation_len, seed):
            results[seed] = load_result(_make_result_path(result_dir, model_hub_path, max_generation_len, seed))
        else:
            results[seed] = {}
    return results


def load_bf16_results(
    benchmark: Benchmark,
    model_name: str,
    max_generation_len: int,
    seeds: Sequence[int],
) -> dict[int, dict]:
    return load_results(make_result_dir("loss", benchmark), make_model_hub_path("bf16", model_name), max_generation_len, seeds)


def save_result(
    result_dir: str,
    model_hub_path: str,
    max_generation_len: int,
    seed: int,
    result: dict,
) -> None:
    path = _make_result_path(result_dir, model_hub_path, max_generation_len, seed)
    HfApi().upload_file(path_or_fileobj=json.dumps(result, indent=2, ensure_ascii=False).encode(), path_in_repo=path, repo_id=REPO_NAME, commit_message=f"save {path}")
    print(f"saved {path}")


def has_tensors(
    result_dir: str,
    model_hub_path: str,
    max_generation_len: int,
    reasoning_seed: int,
) -> bool:
    return HfApi().file_exists(REPO_NAME, _make_tensors_path(result_dir, model_hub_path, max_generation_len, reasoning_seed))


def load_tensors(
    result_dir: str,
    model_hub_path: str,
    max_generation_len: int,
    reasoning_seed: int,
) -> dict[str, Tensor]:
    return load_file(hf_hub_download(REPO_NAME, _make_tensors_path(result_dir, model_hub_path, max_generation_len, reasoning_seed)))


def save_tensors(
    result_dir: str,
    model_hub_path: str,
    max_generation_len: int,
    reasoning_seed: int,
    tensors: dict[str, Tensor],
) -> None:
    path = _make_tensors_path(result_dir, model_hub_path, max_generation_len, reasoning_seed)
    HfApi().upload_file(path_or_fileobj=save(tensors), path_in_repo=path, repo_id=REPO_NAME, commit_message=f"save {path}")
    print(f"saved {path}")
