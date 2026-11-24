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


def extract_model_name(
    model_hub_path: str,
) -> str:
    _, organization, name = model_hub_path.split("/")[:3]
    return f"{organization}/{name}"


def make_result_dir(
    result_name: str,
    benchmark: Benchmark,
) -> str:
    return f"{result_name}/{benchmark}"
