import json
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from transformers import PreTrainedModel

REPO_NAME = "sehyunpark73/multilingual-reasoning-quantization"


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


def _make_result_path(
    experiment_name: str,
    hub_path: str,
    max_generation_len: int,
    seed: int,
) -> str:
    return f"{experiment_name}/{hub_path}/max_generation_len={max_generation_len},seed={seed}.json"


def save_model(
    hub_path: str,
    model: PreTrainedModel,
) -> None:
    with TemporaryDirectory() as directory:
        model.save_pretrained(directory)
        HfApi().upload_folder(folder_path=directory, path_in_repo=hub_path, repo_id=REPO_NAME, commit_message=f"save {hub_path}")
    print(f"saved {hub_path}")


def download_model(
    hub_path: str,
) -> str:
    return str(Path(snapshot_download(REPO_NAME, allow_patterns=[f"{hub_path}/*.json", f"{hub_path}/model*.safetensors"])) / hub_path)


def save_result(
    experiment_name: str,
    hub_path: str,
    max_generation_len: int,
    seed: int,
    result: dict,
) -> None:
    path = _make_result_path(experiment_name, hub_path, max_generation_len, seed)
    HfApi().upload_file(path_or_fileobj=json.dumps(result, indent=2, ensure_ascii=False).encode(), path_in_repo=path, repo_id=REPO_NAME, commit_message=f"save {path}")
    print(f"saved {path}")


def load_results(
    experiment_name: str,
    hub_path: str,
    max_generation_len: int,
    seeds: tuple[int, ...],
) -> dict[int, dict]:
    results = {}
    for seed in seeds:
        path = _make_result_path(experiment_name, hub_path, max_generation_len, seed)
        results[seed] = json.loads(Path(hf_hub_download(REPO_NAME, path)).read_text()) if HfApi().file_exists(REPO_NAME, path) else {}
    return results
