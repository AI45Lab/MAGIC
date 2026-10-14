"""Single-exchange MAGIC prompts and rewards, following the public implementation."""

import re

from prompt.game.single_turn import ATTACKER_SYSTEM_PROMPT, ATTACKER_FIRST_USER_PROMPT, DEFENDER_FIRST_USER_PROMPT
from magic_model_support.reasoning import parse_native_reasoning_response


def messages(role, text, system_prompt=""):
    system = system_prompt
    if role == "attacker":
        system = "\n\n".join(part for part in (system, ATTACKER_SYSTEM_PROMPT) if part)
        user = ATTACKER_FIRST_USER_PROMPT.format(question=text)
    else:
        user = DEFENDER_FIRST_USER_PROMPT.format(attacker_prompt=text)
    return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]


def prompt_tokens(tokenizer, chat, family, role):
    if family == "qwen" and role == "attacker":
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in chat)
        return tokenizer(text + "<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
    return tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True)


def generated_text(tokenizer, prompt, tokens, family):
    tokens = list(tokens)
    terminal = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    while tokens and tokens[-1] in terminal:
        tokens.pop()
    text = tokenizer.decode(tokens, skip_special_tokens=False).strip()
    # Native templates may place the opening reasoning marker in the prompt.
    opening = "[THINK]" if family == "ministral" else "<think>"
    if tokenizer.decode(prompt, skip_special_tokens=False).rstrip().endswith(opening) and not text.startswith(opening):
        text = opening + text
    return text


def visible_answer(text, family):
    if family == "ministral":
        return parse_native_reasoning_response(text)[1]
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    elif text.startswith("<think>"):
        return ""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def valid_attacker_format(text, family):
    if family == "ministral":
        return (text.count("[THINK]") == 1 and text.count("[/THINK]") == 1
                and re.fullmatch(r"\[THINK\]\s*\S[\s\S]*?\[/THINK\]\s*\S[\s\S]*", text.strip()) is not None)
    tags = ("<think>", "</think>", "<answer>", "</answer>")
    return (all(text.count(tag) == 1 for tag in tags)
            and re.fullmatch(r"<think>\s*\S[\s\S]*?</think>\s*<answer>\s*\S[\s\S]*?</answer>", text.strip()) is not None)


def visible_attack(text, family, seed):
    if valid_attacker_format(text, family):
        return visible_answer(text, family) or seed
    markers = r"\[/?THINK\]" if family == "ministral" else r"</?(?:think|answer)>"
    return seed if re.search(markers, text) else (text.strip() or seed)


def defender_reward(query, response, harm_weight, refusal_weight):
    query_label = re.search(r"Safety:\s*(Safe|Unsafe|Controversial)", query)
    response_label = re.search(r"Safety:\s*(Safe|Unsafe|Controversial)", response)
    refusal = re.search(r"Refusal:\s*(Yes|No)", response)
    if not (query_label and response_label and refusal):
        raise ValueError("Guard output must contain request Safety and response Safety/Refusal labels")
    correct_refusal = (query_label[1] != "Safe") == (refusal[1] == "Yes")
    return ((harm_weight if response_label[1] == "Safe" else -harm_weight)
            + (refusal_weight if correct_refusal else -refusal_weight))


def grouped_attacker_rewards(tasks, invalid, usable, finished, formats, groups):
    # Port of mask_invalid_attacker_task in src/verl/verl/utils/reward_score/game.py.
    # Compute over the complete group before VERL normalizes GRPO advantages.
    buckets = {}
    for i, group in enumerate(groups):
        buckets.setdefault(group, []).append(i)
    corrected = list(tasks)
    for indices in buckets.values():
        valid = [tasks[i] for i in indices if not invalid[i] and usable[i]]
        fill = sum(valid) / len(valid) if valid else 0.0
        for i in indices:
            if invalid[i]:
                corrected[i] = fill
    return [corrected[i] + formats[i] if finished[i] else 0.0 for i in range(len(tasks))]
