import logging
import os

import torch
from torch import Tensor
from vllm import LLM, SamplingParams  # type: ignore
from vllm.outputs import CompletionOutput, RequestOutput  # type: ignore
from vllm.tokenizers import TokenizerLike  # type: ignore

from utils.log import log_time
from utils.model import MODEL_CONFIGS

MAX_MODEL_LEN = 40960
ON_POLICY_MAX_MODEL_LEN = 4096
ON_POLICY_GPU_MEMORY_UTILIZATION = 0.25
ON_POLICY_SLEEP_LEVEL = 1


def _load_vllm(
    model_path: str,
    model_name: str,
    max_num_sequences: int | None,
) -> LLM:
    return LLM(model_path, tokenizer=model_name, dtype="bfloat16", max_model_len=MAX_MODEL_LEN, language_model_only=True, max_num_seqs=max_num_sequences)


def load_on_policy_vllm(
    model_name: str,
    seed: int,
) -> LLM:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    vllm_model = LLM(model_name, dtype="bfloat16", max_model_len=ON_POLICY_MAX_MODEL_LEN, language_model_only=True, gpu_memory_utilization=ON_POLICY_GPU_MEMORY_UTILIZATION, enable_sleep_mode=True, seed=seed)
    vllm_model.sleep(level=ON_POLICY_SLEEP_LEVEL)
    logging.getLogger("vllm").setLevel(logging.ERROR)
    return vllm_model


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
    is_reasoning_ids_returned: bool,
) -> list[dict]:
    reasoning_end_id = tokenizer.convert_tokens_to_ids(MODEL_CONFIGS[model_name]["reasoning_end"])
    generations = []
    start = 0
    for input_list in input_lists:
        generation = {"reasoning_lens": [], "is_truncateds": [], "reasonings": [], "responses": []}
        if is_reasoning_ids_returned:
            generation["reasoning_ids"] = []
        for output in outputs[start : start + len(input_list)]:
            completion = output.outputs[0]
            reasoning_len, reasoning, response = _split_completion(completion, tokenizer, reasoning_end_id)
            generation["reasoning_lens"].append(reasoning_len)
            generation["is_truncateds"].append(completion.finish_reason == "length")
            generation["reasonings"].append(reasoning)
            generation["responses"].append(response)
            if is_reasoning_ids_returned:
                generation["reasoning_ids"].append(list(completion.token_ids[:reasoning_len]))
        generations.append(generation)
        start += len(input_list)
    return generations


def _make_chat_inputs(
    model_name: str,
    prompt_lists: list[list[str]],
    prompt_list_seeds: list[int],
    max_generation_len: int,
) -> tuple[list[list[dict]], list[SamplingParams]]:
    messages = []
    sampling_params = []
    for prompts, seed in zip(prompt_lists, prompt_list_seeds):
        for prompt in prompts:
            messages.append([{"role": "user", "content": prompt}])
            sampling_params.append(SamplingParams(**MODEL_CONFIGS[model_name]["sampling"], max_tokens=max_generation_len, seed=seed))
    return messages, sampling_params


def _make_token_inputs(
    model_name: str,
    input_ids_lists: list[list[Tensor]],
    input_ids_list_seeds: list[int],
    max_generation_len: int,
) -> tuple[list[dict], list[SamplingParams]]:
    token_prompts = []
    sampling_params = []
    for input_ids_list, seed in zip(input_ids_lists, input_ids_list_seeds):
        for input_ids in input_ids_list:
            token_prompts.append({"prompt_token_ids": input_ids.tolist()})
            sampling_params.append(SamplingParams(**MODEL_CONFIGS[model_name]["sampling"], max_tokens=max_generation_len, seed=seed))
    return token_prompts, sampling_params


def generate(
    model_path: str,
    model_name: str,
    prompt_lists: list[list[str]],
    prompt_list_seeds: list[int],
    max_generation_len: int,
    max_num_sequences: int | None = None,
    is_reasoning_ids_returned: bool = False,
) -> list[dict]:
    model = _load_vllm(model_path, model_name, max_num_sequences)
    messages, sampling_params = _make_chat_inputs(model_name, prompt_lists, prompt_list_seeds, max_generation_len)
    with log_time(model_path):
        outputs = model.chat(messages, sampling_params, use_tqdm=True, chat_template_kwargs={"enable_thinking": True})
    return _split_outputs(outputs, model.get_tokenizer(), model_name, prompt_lists, is_reasoning_ids_returned)


def generate_without_thinking(
    model_path: str,
    model_name: str,
    prompt_lists: list[list[str]],
    prompt_list_seeds: list[int],
    max_generation_len: int,
) -> list[dict]:
    model = _load_vllm(model_path, model_name, None)
    messages, sampling_params = _make_chat_inputs(model_name, prompt_lists, prompt_list_seeds, max_generation_len)
    with log_time(model_path):
        outputs = model.chat(messages, sampling_params, use_tqdm=True, chat_template_kwargs={"enable_thinking": False})
    generations = []
    start = 0
    for prompts in prompt_lists:
        generation = {"is_truncateds": [], "responses": []}
        for output in outputs[start : start + len(prompts)]:
            completion = output.outputs[0]
            generation["is_truncateds"].append(completion.finish_reason == "length")
            generation["responses"].append(completion.text)
        generations.append(generation)
        start += len(prompts)
    return generations


def continue_generation(
    model_path: str,
    model_name: str,
    input_ids_lists: list[list[Tensor]],
    input_ids_list_seeds: list[int],
    max_generation_len: int,
    is_reasoning_ids_returned: bool = False,
) -> list[dict]:
    model = _load_vllm(model_path, model_name, None)
    token_prompts, sampling_params = _make_token_inputs(model_name, input_ids_lists, input_ids_list_seeds, max_generation_len)
    with log_time(model_path):
        outputs = model.generate(token_prompts, sampling_params, use_tqdm=True)
    return _split_outputs(outputs, model.get_tokenizer(), model_name, input_ids_lists, is_reasoning_ids_returned)


def generate_on_policy(
    vllm_model: LLM,
    model_name: str,
    quantized_weights: list[tuple[str, Tensor]],
    input_ids_list: list[Tensor],
    max_generation_len: int,
) -> dict:
    vllm_model.wake_up()
    vllm_model.collective_rpc("reload_weights", kwargs={"weights_iterator": quantized_weights})
    vllm_model.reset_prefix_cache()
    token_prompts = [{"prompt_token_ids": input_ids.tolist()} for input_ids in input_ids_list]
    sampling_params = SamplingParams(**MODEL_CONFIGS[model_name]["sampling"], max_tokens=max_generation_len)
    outputs = vllm_model.generate(token_prompts, sampling_params, use_tqdm=False)
    vllm_model.sleep(level=ON_POLICY_SLEEP_LEVEL)
    return _split_outputs(outputs, vllm_model.get_tokenizer(), model_name, [input_ids_list], True)[0]


def compute_prompt_logprobs(
    vllm_model: LLM,
    input_ids: Tensor,
) -> Tensor:
    vllm_model.wake_up()
    output = vllm_model.generate([{"prompt_token_ids": input_ids.tolist()}], SamplingParams(max_tokens=1, prompt_logprobs=0), use_tqdm=False)[0]
    vllm_model.sleep(level=ON_POLICY_SLEEP_LEVEL)
    logprobs = []
    for token_id, token_logprobs in zip(input_ids.tolist()[1:], output.prompt_logprobs[1:]):
        logprobs.append(token_logprobs[token_id].logprob)
    return torch.tensor(logprobs)
