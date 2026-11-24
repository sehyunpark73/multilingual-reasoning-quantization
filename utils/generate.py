from vllm import LLM, SamplingParams  # type: ignore
from vllm.outputs import CompletionOutput  # type: ignore
from vllm.tokenizers import TokenizerLike  # type: ignore

from utils.log import log_time

MAX_MODEL_LEN = 40960
GENERATION_CONFIGS = {
    "Qwen/Qwen3.5-2B": {"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}, "reasoning_end": "</think>"},
    "google/gemma-4-E2B-it": {"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "presence_penalty": 0.0}, "reasoning_end": "<channel|>"},
}


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


def generate(
    model_path: str,
    model_name: str,
    prompt_lists: list[list[str]],
    prompt_list_seeds: list[int],
    max_generation_len: int,
) -> list[dict]:
    model = LLM(model_path, tokenizer=model_name, dtype="bfloat16", max_model_len=MAX_MODEL_LEN, language_model_only=True)
    messages = []
    sampling_params = []
    for prompts, seed in zip(prompt_lists, prompt_list_seeds):
        for prompt in prompts:
            messages.append([{"role": "user", "content": prompt}])
            sampling_params.append(SamplingParams(**GENERATION_CONFIGS[model_name]["sampling"], max_tokens=max_generation_len, seed=seed))
    with log_time(model_path):
        outputs = model.chat(messages, sampling_params, use_tqdm=True, chat_template_kwargs={"enable_thinking": True})

    tokenizer = model.get_tokenizer()
    reasoning_end_id = tokenizer.convert_tokens_to_ids(GENERATION_CONFIGS[model_name]["reasoning_end"])
    generations = []
    start = 0
    for prompts in prompt_lists:
        generation = {"reasoning_lens": [], "truncateds": [], "reasonings": [], "responses": []}
        for output in outputs[start : start + len(prompts)]:
            completion = output.outputs[0]
            reasoning_len, reasoning, response = _split_completion(completion, tokenizer, reasoning_end_id)
            generation["reasoning_lens"].append(reasoning_len)
            generation["truncateds"].append(completion.finish_reason == "length")
            generation["reasonings"].append(reasoning)
            generation["responses"].append(response)
        generations.append(generation)
        start += len(prompts)
    return generations
