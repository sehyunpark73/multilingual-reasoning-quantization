from utils.dataset import Benchmark


def make_model_hub_path(
    method: str,
    model_name: str,
    settings: dict[str, object] | None = None,
) -> str:
    if settings is None:
        return f"{method}/{model_name}"
    setting_texts = [f"{name}={value}" for name, value in settings.items()]
    settings_text = ",".join(setting_texts)
    return f"{method}/{model_name}/{settings_text}"


def make_quantized_finetune_model_hub_path(
    finetune_model_hub_path: str,
    method: str,
) -> str:
    finetune_name, model_name_and_settings = finetune_model_hub_path.split("/", 1)
    return f"{finetune_name}_{method}/{model_name_and_settings}"


def extract_model_name(
    model_hub_path: str,
) -> str:
    _, organization, name = model_hub_path.split("/")[:3]
    return f"{organization}/{name}"


def extract_method(
    model_hub_path: str,
) -> str:
    return model_hub_path.split("/")[0]


def make_result_dir(
    result_name: str,
    benchmark: Benchmark,
) -> str:
    return f"{result_name}/{benchmark}"


def make_reasoning_prefix_result_dir(
    length_ratio: float,
) -> str:
    result_dir = make_result_dir("reasoning_prefix_insert", "mgsm")
    return f"{result_dir}/length_ratio={length_ratio}"
