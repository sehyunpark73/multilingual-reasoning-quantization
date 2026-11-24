from collections import defaultdict
from typing import Literal, TypeAlias

from datasets import load_dataset
import torch
from torch import Tensor
from transformers import AutoTokenizer

LANGUAGES = ["en", "zh", "ja", "sw", "te"]
NON_EN_LANGUAGES = [language for language in LANGUAGES if language != "en"]
MGSM_PROMPTS = {
    "en": """Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of "Answer:". Do not add anything other than the integer answer after "Answer:".\n\n{question}""",
    "zh": """解决这个数学问题。在最后一行给出答案前，请提供推理步骤。最后一行应该以 "答案: " 的形式独立给出答案。在 "答案：" 后不要添加除整数答案之外的任何内容。\n\n{question}""",
    "ja": """の数学の問題を解いてください。最終的な答えを出す前に、解答の推論過程を記述してください。そして最後の行には "答え:" の形式で答えを記述し、その後には整数の答え以外何も追加しないでください。\n\n{question}""",
    "sw": """Suluhisha tatizo hili la hesabu. Toa hatua za mantiki kabla ya kutoa jibu la mwisho kwenye mstari wa mwisho peke yake katika muundo wa "Jibu:". Usiongeze chochote kingine isipokuwa jibu la integer baada ya "Jibu:".\n\n{question}""",
    "te": """ఈ గణిత సమస్యను పరిష్కరించండి. చివరి సమాధానాన్ని ఇవ్వదానికి ముందు తర్కాత్మక అదుగులను ఇవ్వండి. చివరి పంక్తిలో మాత్రమే 'సమాధానం:' అనే ఆకారంలో చివరి సమాధానాద్ని ఇవ్వండి సమాధానం: తర్వాత పూర్ణాంక సమాధానానికి తప్పించి ఎదేనా చేర్చవద్దు.\n\n{question}""",
}
MMLU_PROX_LITE_TEMPLATES = {
    "en": {"question_label": "Question:", "options_label": "Options:", "instruction": 'The following are multiple choice questions (with answers) about {subject}. Think step by step and then finish your answer with "{answer_format}" where X is the correct letter choice.', "answer_format": "the answer is ({})"},
    "zh": {"question_label": "问题：", "options_label": "选项：", "instruction": '以下是关于{subject}的选择题（带有答案）。请逐步思考，然后以"{answer_format}"结束您的回答，其中X是正确的选项字母。', "answer_format": "答案是 ({})"},
    "ja": {"question_label": "質問：", "options_label": "選択肢：", "instruction": "以下は{subject}に関する選択問題（解答付き）です。段階的に考え、最後に「{answer_format}」と回答を締めくくってください。Xは正解の選択肢を示す文字です。", "answer_format": "答えは ({}) です"},
    "sw": {"question_label": "Swali:", "options_label": "Chaguo:", "instruction": 'Yafuatayo ni maswali ya chaguo-nyingi (yenye majibu) kuhusu {subject}. Fikiria hatua kwa hatua kisha malizia jibu lako kwa "{answer_format}" ambapo X ni herufi ya chaguo sahihi.', "answer_format": "Jibu ni ({})"},
    "te": {"question_label": "ప్రశ్న:", "options_label": "ఎంపికలు:", "instruction": 'క్రింది {subject}కి సంబంధించిన బహుళఎంపిక ప్రశ్న (సమాధానాలతో). దయచేసి దశలవారీగా ఆలోచించి, మీ సమాధానాన్ని "{answer_format}"తో ముగించండి, ఇక్కడ X సరైన ఎంపిక అక్షరం.', "answer_format": "సమాధానం ({})"},
}
MMLU_PROX_LITE_SUBJECTS = {
    "en": {"biology": "biology", "business": "business", "chemistry": "chemistry", "computer science": "computer_science", "economics": "economics", "engineering": "engineering", "health": "health", "history": "history", "law": "law", "math": "math", "other": "other", "philosophy": "philosophy", "physics": "physics", "psychology": "psychology"},
    "zh": {"biology": "生物学", "business": "商业", "chemistry": "化学", "computer science": "计算机科学", "economics": "经济学", "engineering": "工程学", "health": "健康", "history": "历史", "law": "法律", "math": "数学", "other": "其他", "philosophy": "哲学", "physics": "物理学", "psychology": "心理学"},
    "ja": {"biology": "生物学", "business": "ビジネス", "chemistry": "化学", "computer science": "コンピュータサイエンス", "economics": "経済学", "engineering": "工学", "health": "健康科学", "history": "歴史", "law": "法律", "math": "数学", "other": "その他", "philosophy": "哲学", "physics": "物理学", "psychology": "心理学"},
    "sw": {"biology": "biolojia", "business": "biashara", "chemistry": "kemia", "computer science": "sayansi ya kompyuta", "economics": "uchumi", "engineering": "uhandisi", "health": "afya", "history": "historia", "law": "sheria", "math": "hisabati", "other": "nyingine", "philosophy": "falsafa", "physics": "fizikia", "psychology": "saikolojia"},
    "te": {"biology": "జీవశాస్త్రం", "business": "వ్యాపారం", "chemistry": "రసాయన శాస్త్రం", "computer science": "కంప్యూటర్ సైన్స్", "economics": "ఆర్థిక శాస్త్రం", "engineering": "ఇంజనీరింగ్", "health": "ఆరోగ్యం", "history": "చరిత్ర", "law": "న్యాయశాస్త్రం", "math": "గణితం", "other": "ఇతరమైన", "philosophy": "తత్వవేత్త", "physics": "భౌతిక శాస్త్రం", "psychology": "మనోవిజ్ఞానశాస్త్రం"},
}
MAX_NUM_OPTIONS = 10
SHUFFLE_BUFFER_SIZE = 10000
MCOT_BATCH_SIZE = 100000

Benchmark: TypeAlias = Literal["mgsm", "mmlu_prox_lite"]


def load_wikipedia(
    model_name: str,
    calibration_languages: tuple[str, ...],
    num_chunks: int,
    chunk_len: int,
    shuffle_seed: int,
) -> list[Tensor]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bos_ids = [] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id]
    text_len = chunk_len - len(bos_ids)
    chunks = []
    for language_index, language in enumerate(calibration_languages):
        num_language_chunks = num_chunks // len(calibration_languages)
        if language_index < num_chunks % len(calibration_languages):
            num_language_chunks += 1
        dataset = load_dataset("wikimedia/wikipedia", f"20231101.{language}", split="train", streaming=True).shuffle(seed=shuffle_seed, buffer_size=SHUFFLE_BUFFER_SIZE)
        token_ids = []
        language_chunks = []
        for row in dataset:
            token_ids.extend(tokenizer(row["text"] + "\n\n", add_special_tokens=False)["input_ids"])
            while len(token_ids) >= text_len and len(language_chunks) < num_language_chunks:
                language_chunks.append(torch.tensor(bos_ids + token_ids[:text_len]))
                token_ids = token_ids[text_len:]
            if len(language_chunks) == num_language_chunks:
                break
        chunks.extend(language_chunks)
    return chunks


def load_mgsm() -> dict[str, dict[str, list]]:
    problems = {language: {"questions": [], "prompts": [], "answers": []} for language in LANGUAGES}
    for language in LANGUAGES:
        for row in load_dataset("juletxara/mgsm", language, split="test"):
            problems[language]["questions"].append(row["question"])
            problems[language]["prompts"].append(MGSM_PROMPTS[language].format(question=row["question"]))
            problems[language]["answers"].append(row["answer_number"])
    return problems


def load_mmlu_prox_lite() -> dict[str, dict[str, list]]:
    problems = {language: {"prompts": [], "answers": []} for language in LANGUAGES}
    for language in LANGUAGES:
        template = MMLU_PROX_LITE_TEMPLATES[language]
        for row in load_dataset("li-lab/MMLU-ProX-Lite", language, split="test"):
            prompt = template["instruction"].format(subject=MMLU_PROX_LITE_SUBJECTS[language][row["category"]], answer_format=template["answer_format"].format("X"))
            prompt += f"\n\n{template['question_label']}\n{row['question']}\n{template['options_label']}\n"
            options = [row[f"option_{option_index}"] for option_index in range(MAX_NUM_OPTIONS) if row[f"option_{option_index}"] is not None]
            for option_index, option in enumerate(options):
                prompt += f"{chr(ord('A') + option_index)}. {option}\n"
            problems[language]["prompts"].append(prompt)
            problems[language]["answers"].append(row["answer"])
    return problems


def _load_mcot_translations() -> dict[str, dict[str, str]]:
    mcot = load_dataset("laihuiyuan/mCoT-MATH", data_files="train_part_*.jsonl", split="train").select_columns(["question", "lang-id"])
    translations_by_mcot_id = defaultdict(dict)
    for batch in mcot.iter(batch_size=MCOT_BATCH_SIZE):
        for question, language_and_mcot_id in zip(batch["question"], batch["lang-id"]):
            language, mcot_id = language_and_mcot_id.split("-")
            if language in LANGUAGES:
                translations_by_mcot_id[mcot_id][language] = question.strip()
    return translations_by_mcot_id


def load_translated_gsm8k() -> dict[str, dict[str, list]]:
    translations_by_en_question = {}
    for translations in _load_mcot_translations().values():
        en_question = " ".join(translations["en"].split())
        translations_by_en_question[en_question] = translations
    problems = {language: {"questions": [], "prompts": [], "answers": []} for language in LANGUAGES}
    for row in load_dataset("openai/gsm8k", "main", split="train"):
        translations = translations_by_en_question[" ".join(row["question"].split())]
        answer = int(row["answer"].split("####")[-1].replace(",", ""))
        for language in LANGUAGES:
            problems[language]["questions"].append(translations[language])
            problems[language]["prompts"].append(MGSM_PROMPTS[language].format(question=translations[language]))
            problems[language]["answers"].append(answer)
    return problems
