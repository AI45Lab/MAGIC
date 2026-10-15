"""Modern VERL AgentLoop implementation of MAGIC's two-role rollout."""

from __future__ import annotations

import asyncio
import os
import re
import time
from typing import Any
from uuid import uuid4

import httpx

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

from magic_model_support.rl.contracts import (
    attacker_messages,
    attacker_template_from_raw_prompt,
    defender_messages,
    format_reward,
    mistral_reasoning_api_messages,
    parse_native_reasoning_response,
    safety_reward,
    seed_prompt_from_sample,
    valid_attacker_format,
    visible_attack as get_visible_attack,
)


_CONTEXT_OVERFLOW_RE = re.compile(
    r"maximum context length is (?P<context>\d+) tokens.*?"
    r"requested (?P<output>\d+) output tokens.*?"
    r"prompt contains at least (?P<input>\d+) input tokens",
    re.IGNORECASE | re.DOTALL,
)


_CONTEXT_TOKENIZATION_SAFETY_MARGIN = 8


async def _server_chat_prompt_tokens(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    timeout_s: float,
) -> tuple[int, int | None]:
    """Ask vLLM's own renderer for the exact chat prompt length.

    The active VERL tokenizer and the frozen vLLM endpoint can render the same
    chat with slightly different special-token boundaries.  The server's
    ``/tokenize`` route uses the exact renderer that later validates the chat
    completion request, so it is the source of truth for context capping.
    """

    server_root = base_url.rstrip("/")
    if server_root.endswith("/v1"):
        server_root = server_root[:-3]
    async with httpx.AsyncClient(timeout=min(timeout_s, 60.0)) as client:
        response = await client.post(
            f"{server_root}/tokenize",
            headers={"Authorization": "Bearer local-placeholder"},
            json={
                "model": model,
                "messages": messages,
                "add_generation_prompt": True,
            },
        )
        response.raise_for_status()
        body = response.json()
    count = int(body["count"])
    if count < 0:
        raise ValueError(f"vLLM /tokenize returned a negative count: {count}")
    raw_max_model_len = body.get("max_model_len")
    max_model_len = (
        int(raw_max_model_len) if raw_max_model_len is not None else None
    )
    return count, max_model_len


def _context_limited_max_tokens(
    response: httpx.Response, requested_max_tokens: int
) -> int | None:
    """Return a valid output cap for vLLM context-overflow 400 responses."""

    if response.status_code != 400:
        return None
    match = _CONTEXT_OVERFLOW_RE.search(response.text)
    if match is None:
        return None
    remaining = int(match.group("context")) - int(match.group("input"))
    safe_remaining = remaining - _CONTEXT_TOKENIZATION_SAFETY_MARGIN
    if 0 < safe_remaining < requested_max_tokens:
        return safe_remaining
    return None


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required for MAGIC RL")
    return value.rstrip("/")


async def _chat_completion(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_s: float,
    max_retries: int,
    preserve_special_tokens: bool = False,
) -> tuple[str, str | None]:
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if preserve_special_tokens:
        payload["skip_special_tokens"] = False
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                response = await client.post(
                    url,
                    headers={"Authorization": "Bearer local-placeholder"},
                    json=payload,
                )
                response.raise_for_status()
                body = response.json()
            choice = body["choices"][0]
            return (choice.get("message", {}).get("content") or "", choice.get("finish_reason"))
        except httpx.HTTPStatusError as exc:
            last_error = exc
            context_limited_max_tokens = _context_limited_max_tokens(
                exc.response, int(payload["max_tokens"])
            )
            if context_limited_max_tokens is not None and attempt < max_retries:
                print(
                    "MAGIC external generation context cap: "
                    f"max_tokens {payload['max_tokens']} -> "
                    f"{context_limited_max_tokens}",
                    flush=True,
                )
                payload["max_tokens"] = context_limited_max_tokens
                continue
            if attempt < max_retries:
                await asyncio.sleep(min(2**attempt, 8))
        except Exception as exc:  # retry only around the full HTTP transaction
            last_error = exc
            if attempt < max_retries:
                await asyncio.sleep(min(2**attempt, 8))
    raise RuntimeError(f"chat completion failed after {max_retries + 1} attempts: {last_error}")


class MagicGameAgentLoop(AgentLoopBase):
    """One attacker/defender game while optimizing exactly one active role.

    The active role is generated by VERL's managed vLLM server so its token IDs and
    log probabilities remain trainable.  The frozen opponent and Qwen3Guard are
    called through localhost OpenAI-compatible endpoints on reserved GPUs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_role = _required_env("MAGIC_TRAIN_ROLE").lower()
        if self.train_role not in {"attacker", "defender"}:
            raise ValueError(f"invalid MAGIC_TRAIN_ROLE={self.train_role!r}")
        self.opponent_endpoint = _required_env("MAGIC_OPPONENT_ENDPOINT")
        self.opponent_model = _required_env("MAGIC_OPPONENT_MODEL")
        self.guard_endpoint = _required_env("WILDGUARD_API_ENDPOINT")
        self.guard_model = _required_env("WILDGUARD_MODEL_NAME")
        self.response_length = int(self.rollout_config.response_length)
        self.opponent_max_model_len = int(os.environ.get("MAX_MODEL_LEN", "0"))
        self.http_timeout_s = float(os.environ.get("MAGIC_HTTP_TIMEOUT_S", "900"))
        self.http_retries = int(os.environ.get("MAGIC_HTTP_MAX_RETRIES", "2"))
        self.opponent_max_resamples = int(
            os.environ["MAGIC_OPPONENT_MAX_RESAMPLES"]
        )
        if self.opponent_max_resamples < 0:
            raise ValueError("MAGIC_OPPONENT_MAX_RESAMPLES must be non-negative")
        self.guard_max_tokens = int(os.environ.get("REWARD_MAX_TOKENS", "256"))
        self.harm_weight = float(os.environ.get("REWARD_HARM", "1.0"))
        self.refusal_weight = float(os.environ.get("REWARD_REFUSAL", "0.5"))
        self.format_weight = float(os.environ.get("FORMAT_REWARD_VALUE", "0.5"))
        self.native_reasoning = os.environ.get("MAGIC_NATIVE_REASONING", "0") == "1"

    async def _external_generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
    ) -> tuple[str, str | None, dict[str, Any]]:
        external_messages = (
            messages
            if self.native_reasoning
            else mistral_reasoning_api_messages(messages)
        )
        request_max_tokens = self.response_length
        opponent_prompt_tokens: int | None = None
        opponent_prompt_token_source = "unavailable"
        if getattr(self, "opponent_max_model_len", 0) > 0:
            prompt_token_ids = self.tokenizer.apply_chat_template(
                external_messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            opponent_prompt_tokens = len(prompt_token_ids)
            opponent_prompt_token_source = "local_fallback"
            effective_max_model_len = self.opponent_max_model_len
            try:
                (
                    server_prompt_tokens,
                    server_max_model_len,
                ) = await _server_chat_prompt_tokens(
                    base_url=self.opponent_endpoint,
                    model=self.opponent_model,
                    messages=external_messages,
                    timeout_s=self.http_timeout_s,
                )
                opponent_prompt_tokens = server_prompt_tokens
                opponent_prompt_token_source = "vllm_tokenize"
                if server_max_model_len is not None and server_max_model_len > 0:
                    effective_max_model_len = min(
                        effective_max_model_len, server_max_model_len
                    )
            except Exception as exc:
                print(
                    "MAGIC external generation exact tokenization unavailable; "
                    f"using local fallback; error={exc!r}",
                    flush=True,
                )
            remaining_context = (
                effective_max_model_len
                - opponent_prompt_tokens
                - _CONTEXT_TOKENIZATION_SAFETY_MARGIN
            )
            if remaining_context <= 0:
                print(
                    "MAGIC external generation skipped: prompt consumes context; "
                    f"prompt_tokens={opponent_prompt_tokens}; "
                    f"max_model_len={effective_max_model_len}; "
                    f"token_source={opponent_prompt_token_source}",
                    flush=True,
                )
                return "", "length", {
                    "attempts": 0,
                    "resamples": 0,
                    "reasoning_chars": 0,
                    "reasoning_issue": "prompt_exceeds_context",
                    "opponent_prompt_tokens": opponent_prompt_tokens,
                    "opponent_max_tokens": 0,
                    "opponent_prompt_token_source": opponent_prompt_token_source,
                }
            request_max_tokens = min(self.response_length, remaining_context)
            if request_max_tokens < self.response_length:
                print(
                    "MAGIC external generation preflight context cap: "
                    f"max_tokens {self.response_length} -> {request_max_tokens}; "
                    f"prompt_tokens={opponent_prompt_tokens}; "
                    f"max_model_len={effective_max_model_len}; "
                    f"token_source={opponent_prompt_token_source}",
                    flush=True,
                )
        last_finish_reason: str | None = None
        last_diagnostic: dict[str, Any] = {}
        for resamples in range(self.opponent_max_resamples + 1):
            raw_text, finish_reason = await _chat_completion(
                base_url=self.opponent_endpoint,
                model=self.opponent_model,
                messages=external_messages,
                temperature=temperature,
                top_p=1.0,
                max_tokens=request_max_tokens,
                timeout_s=self.http_timeout_s,
                max_retries=self.http_retries,
                preserve_special_tokens=self.native_reasoning,
            )
            if not self.native_reasoning:
                return raw_text, finish_reason, {
                    "attempts": resamples + 1,
                    "resamples": resamples,
                    "reasoning_chars": 0,
                    "reasoning_issue": None,
                    "opponent_prompt_tokens": opponent_prompt_tokens,
                    "opponent_max_tokens": request_max_tokens,
                    "opponent_prompt_token_source": opponent_prompt_token_source,
                }

            reasoning, visible, issue = parse_native_reasoning_response(raw_text)
            last_finish_reason = finish_reason
            last_diagnostic = {
                "raw_output": raw_text,
                "attempts": resamples + 1,
                "resamples": resamples,
                "reasoning_chars": len(reasoning),
                "reasoning_issue": issue,
                "opponent_prompt_tokens": opponent_prompt_tokens,
                "opponent_max_tokens": request_max_tokens,
                "opponent_prompt_token_source": opponent_prompt_token_source,
            }
            if visible:
                return visible, finish_reason, last_diagnostic
            if resamples < self.opponent_max_resamples:
                print(
                    "MAGIC opponent produced no visible response; "
                    f"resampling {resamples + 1}/{self.opponent_max_resamples}; "
                    f"issue={issue}; finish_reason={finish_reason}",
                    flush=True,
                )
        return "", last_finish_reason, last_diagnostic

    async def _guard(self, attacker_prompt: str, defender_response: str):
        query_task = _chat_completion(
            base_url=self.guard_endpoint,
            model=self.guard_model,
            messages=[{"role": "user", "content": attacker_prompt}],
            temperature=0.0,
            top_p=1.0,
            max_tokens=self.guard_max_tokens,
            timeout_s=self.http_timeout_s,
            max_retries=self.http_retries,
        )
        response_task = _chat_completion(
            base_url=self.guard_endpoint,
            model=self.guard_model,
            messages=[
                {"role": "user", "content": attacker_prompt},
                {"role": "assistant", "content": defender_response},
            ],
            temperature=0.0,
            top_p=1.0,
            max_tokens=self.guard_max_tokens,
            timeout_s=self.http_timeout_s,
            max_retries=self.http_retries,
        )
        (query_text, _), (response_text, _) = await asyncio.gather(query_task, response_task)
        return query_text, response_text

    async def _active_generate(
        self,
        messages: list[dict[str, str]],
        sampling_params: dict[str, Any],
        priority: int,
    ):
        prompt_ids = await self.ct_build_initial_tokens(messages)
        request_id = (
            f"det-{priority}"
            if getattr(self.rollout_config, "full_determinism", False)
            else uuid4().hex
        )
        token_output: TokenOutput = await self.server_manager.generate(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            priority=priority,
        )
        merge_result, response_mask, response_logprobs = await self.ct_merge_assistant_token(
            prompt_ids,
            token_output.token_ids,
            [],
            [] if token_output.log_probs else None,
            assistant_logprobs=token_output.log_probs if token_output.log_probs else None,
        )
        response_ids = merge_result.token_ids[-len(response_mask) :] if response_mask else []
        final_prompt_ids = merge_result.token_ids[: len(merge_result.token_ids) - len(response_mask)]
        active_text = self.tokenizer.decode(
            token_output.token_ids, skip_special_tokens=not self.native_reasoning
        )
        return (
            final_prompt_ids,
            response_ids[: self.response_length],
            response_mask[: self.response_length],
            response_logprobs[: self.response_length] if response_logprobs else None,
            active_text,
            token_output,
        )

    async def run(
        self, sampling_params: dict[str, Any], priority: int = 0, **kwargs
    ) -> AgentLoopOutput:
        priority = int(priority)
        attacker_template = attacker_template_from_raw_prompt(kwargs["raw_prompt"])
        seed_prompt = seed_prompt_from_sample(kwargs)
        data_type = str(kwargs.get("data_type") or (kwargs.get("extra_info") or {}).get("data_type") or "")
        metrics: dict[str, Any] = {}
        started = time.monotonic()
        attacker_reasoning_issue: str | None = None
        defender_reasoning_issue: str | None = None
        opponent_resamples = 0
        zero_reward_reason: str | None = None

        if self.train_role == "defender":
            # Official MAGIC makes the frozen attacker deterministic in defender blocks.
            with simple_timer("opponent_generate", metrics):
                attacker_output, attacker_finish, opponent_diagnostic = await self._external_generate(
                    attacker_messages(attacker_template), temperature=0.0
                )
            attacker_reasoning_issue = opponent_diagnostic.get("reasoning_issue")
            opponent_resamples = int(opponent_diagnostic.get("resamples", 0))
            attacker_output = opponent_diagnostic.get("raw_output", attacker_output)
            visible_attack = get_visible_attack(
                attacker_output, seed_prompt, native_reasoning=self.native_reasoning
            )
            if not visible_attack:
                zero_reward_reason = "frozen_attacker_no_visible_response"
            active_messages = defender_messages(visible_attack)
            with simple_timer("generate_sequences", metrics):
                active = await self._active_generate(active_messages, sampling_params, priority)
            defender_output = active[4]
            defender_finish = active[5].stop_reason
            if self.native_reasoning:
                _, visible_defender, defender_reasoning_issue = (
                    parse_native_reasoning_response(defender_output)
                )
            else:
                visible_defender = defender_output
            if not visible_defender:
                zero_reward_reason = zero_reward_reason or "active_defender_no_visible_response"
            format_component = 0.0
        else:
            active_messages = attacker_messages(attacker_template)
            with simple_timer("generate_sequences", metrics):
                active = await self._active_generate(active_messages, sampling_params, priority)
            attacker_output = active[4]
            attacker_finish = active[5].stop_reason
            if self.native_reasoning:
                _, _, attacker_reasoning_issue = (
                    parse_native_reasoning_response(attacker_output)
                )
            visible_attack = get_visible_attack(
                attacker_output, seed_prompt, native_reasoning=self.native_reasoning
            )
            if not visible_attack:
                zero_reward_reason = "active_attacker_no_visible_response"
                defender_output = ""
                visible_defender = ""
                defender_finish = None
            else:
                # Official MAGIC keeps the frozen defender sampled in attacker blocks.
                with simple_timer("opponent_generate", metrics):
                    defender_output, defender_finish, opponent_diagnostic = (
                        await self._external_generate(
                            defender_messages(visible_attack), temperature=1.0
                        )
                    )
                defender_reasoning_issue = opponent_diagnostic.get("reasoning_issue")
                opponent_resamples = int(opponent_diagnostic.get("resamples", 0))
                visible_defender = defender_output
                if not visible_defender:
                    zero_reward_reason = "frozen_defender_no_visible_response"
            format_component = format_reward(
                attacker_output, self.format_weight, native_reasoning=self.native_reasoning
            )

        attacker_invalid = not valid_attacker_format(
            attacker_output, native_reasoning=self.native_reasoning
        )
        if zero_reward_reason is None:
            with simple_timer("compute_score", metrics):
                query_guard, response_guard = await self._guard(
                    visible_attack, visible_defender
                )
                components = safety_reward(
                    query_guard,
                    response_guard,
                    data_type=data_type,
                    harm_weight=self.harm_weight,
                    refusal_weight=self.refusal_weight,
                )
            reward_harm = components.reward_harm
            reward_refusal = components.reward_refusal
            request_label = components.request_label
            response_label = components.response_label
            refusal_label = components.refusal_label
            task_reward = (
                components.total
                if self.train_role == "defender"
                else -components.total
            )
        else:
            task_reward = 0.0
            reward_harm = 0.0
            reward_refusal = 0.0
            request_label = "SkippedNoVisibleResponse"
            response_label = "SkippedNoVisibleResponse"
            refusal_label = "SkippedNoVisibleResponse"

        # The manager replaces invalid attacker task rewards with the group's
        # valid mean after all workers finish. Never suppress the format penalty.
        reward = task_reward + format_component
        token_output = active[5]
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = token_output.num_preempted if token_output.num_preempted is not None else -1
        metrics["magic_total_seconds"] = time.monotonic() - started
        metrics["opponent_resamples"] = opponent_resamples
        metrics["zero_reward_no_visible_response"] = (
            1.0 if zero_reward_reason is not None else 0.0
        )
        extra_fields = dict(token_output.extra_fields or {})
        extra_fields.update(
            {
                "magic_train_role": self.train_role,
                "attacker_output": attacker_output,
                "visible_attacker_output": visible_attack,
                "attacker_format_invalid": attacker_invalid,
                "magic_raw_task_reward": task_reward,
                "magic_task_reward": task_reward,
                "magic_task_reward_usable": zero_reward_reason is None,
                "defender_output": defender_output,
                "visible_defender_output": visible_defender,
                "attacker_finish_reason": attacker_finish,
                "defender_finish_reason": defender_finish,
                "attacker_reasoning_issue": attacker_reasoning_issue,
                "defender_reasoning_issue": defender_reasoning_issue,
                "opponent_resamples": opponent_resamples,
                "zero_reward_reason": zero_reward_reason,
                "reward_harm": reward_harm,
                "reward_refusal": reward_refusal,
                "format_reward": format_component,
                "request_safety_label": request_label,
                "response_safety_label": response_label,
                "response_refusal_label": refusal_label,
                "turn_scores": [],
                "tool_rewards": [],
            }
        )
        return AgentLoopOutput(
            prompt_ids=active[0],
            response_ids=active[1],
            response_mask=active[2],
            response_logprobs=active[3],
            routed_experts=(
                token_output.routed_experts[: len(active[0]) + self.response_length]
                if token_output.routed_experts is not None
                else None
            ),
            reward_score=reward,
            num_turns=2,
            metrics=metrics,
            extra_fields=extra_fields,
        )
