from torch import Tensor
from vllm import LLM, SamplingParams  # type: ignore
from vllm.outputs import CompletionOutput, RequestOutput  # type: ignore
from vllm.tokenizers import TokenizerLike  # type: ignore

from utils.log import log_time
from utils.model import MODEL_CONFIGS

MAX_MODEL_LEN = 40960


def _load_vllm(
    model_path: str,
    model_name: str,
    max_num_sequences: int | None,
) -> LLM:
    return LLM(model_path, tokenizer=model_name, dtype="bfloat16", max_model_len=MAX_MODEL_LEN, language_model_only=True, max_num_seqs=max_num_sequences)


def _split_completion(
    completion: CompletionOutput,
    tokenizer: TokenizerLike,
    reasoning_end_id: int,
) -> tuple[int, str, str | None]:
    token_ids = list(completion.token_ids)
    if reasoning_end_id not in token_ids:
        return len(token_ids), tokenizer.decode(token_ids, skip_special_tokens=True), None
    reasoning_len = token_ids.index(reasoning_end_id)
    return reasoning_len, tokenizer.decode(token_ids[:reasoning_len], skip_special_tokens=True), tokenizer.decode(token_ids[reasoning_len + 1 :], skip_special_tokens=True)


def _split_outputs(
    outputs: list[RequestOutput],
    tokenizer: TokenizerLike,
    model_name: str,
    input_lists: list[list],
) -> list[dict]:
    reasoning_end_id = tokenizer.convert_tokens_to_ids(MODEL_CONFIGS[model_name]["reasoning_end"])
    generations = []
    start = 0
    for input_list in input_lists:
        generation = {"reasoning_lens": [], "is_truncateds": [], "reasonings": [], "responses": []}
        for output in outputs[start : start + len(input_list)]:
            completion = output.outputs[0]
            reasoning_len, reasoning, response = _split_completion(completion, tokenizer, reasoning_end_id)
            generation["reasoning_lens"].append(reasoning_len)
            generation["is_truncateds"].append(completion.finish_reason == "length")
            generation["reasonings"].append(reasoning)
            generation["responses"].append(response)
        generations.append(generation)
        start += len(input_list)
    return generations


def generate(
    model_path: str,
    model_name: str,
    prompt_lists: list[list[str]],
    prompt_list_seeds: list[int],
    max_generation_len: int,
    max_num_sequences: int | None = None,
) -> list[dict]:
    model = _load_vllm(model_path, model_name, max_num_sequences)
    messages = []
    sampling_params = []
    for prompts, seed in zip(prompt_lists, prompt_list_seeds):
        for prompt in prompts:
            messages.append([{"role": "user", "content": prompt}])
            sampling_params.append(SamplingParams(**MODEL_CONFIGS[model_name]["sampling"], max_tokens=max_generation_len, seed=seed))
    with log_time(model_path):
        outputs = model.chat(messages, sampling_params, use_tqdm=True, chat_template_kwargs={"enable_thinking": True})
    return _split_outputs(outputs, model.get_tokenizer(), model_name, prompt_lists)


def continue_generation(
    model_path: str,
    model_name: str,
    input_ids_lists: list[list[Tensor]],
    input_ids_list_seeds: list[int],
    max_generation_len: int,
) -> list[dict]:
    model = _load_vllm(model_path, model_name, None)
    token_prompts = []
    sampling_params = []
    for input_ids_list, seed in zip(input_ids_lists, input_ids_list_seeds):
        for input_ids in input_ids_list:
            token_prompts.append({"prompt_token_ids": input_ids.tolist()})
            sampling_params.append(SamplingParams(**MODEL_CONFIGS[model_name]["sampling"], max_tokens=max_generation_len, seed=seed))
    with log_time(model_path):
        outputs = model.generate(token_prompts, sampling_params, use_tqdm=True)
    return _split_outputs(outputs, model.get_tokenizer(), model_name, input_ids_lists)
