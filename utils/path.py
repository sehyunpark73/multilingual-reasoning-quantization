from utils.dataset import Benchmark

RESULT_DIRS = {
    "position_kl": "position_kl/mgsm",
    "error_types": "error_types/mgsm",
    "understanding_insert": "understanding_insert/mgsm",
    "en_understanding_insert": "en_understanding_insert/mgsm",
    "reasoning_prefix_insert": "reasoning_prefix_insert/mgsm",
    "en_question_insert": "en_question_insert/mgsm",
    "bf16_translation_insert": "bf16_translation_insert/mgsm",
    "translate_first": "translate_first/mgsm",
    "mgsm_translation": "mgsm_translation",
    "gsm8k_reasoning": "gsm8k_reasoning",
    "gsm8k_en_question_insert_reasoning": "gsm8k_en_question_insert_reasoning",
}


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
    setting_texts = [f"{name}={value}" for name, value in settings.items()]
    settings_text = ",".join(setting_texts)
    return f"{finetune_name}/{model_name}/{settings_text}"


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
