"""Modern VERL AgentLoop for MAGIC's public single-exchange game."""

import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.utils.profiler import simple_timer

from magic_model_support.http_client import request
from magic_model_support.training.protocol import (
    messages, prompt_tokens, generated_text, visible_answer, visible_attack, valid_attacker_format, defender_reward,
)


class MagicAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.settings = json.loads(Path(os.environ["MAGIC_RUN_CONFIG"]).read_text())
        self.role = os.environ["MAGIC_TRAIN_ROLE"]
        self.family = self.settings["model_family"]
        path = self.settings.get("system_prompt_file")
        self.system_prompt = Path(path).read_text().strip() if path else ""

    def _prompt(self, role, text):
        ids = prompt_tokens(self.tokenizer, messages(role, text, self.system_prompt), self.family, role)
        if len(ids) > self.rollout_config.prompt_length:
            raise ValueError(f"{role} prompt exceeds rollout.prompt_length; increase the configured context budget")
        return ids

    async def _opponent(self, role, text):
        ids = self._prompt(role, text)
        result = await request(self.settings["opponent_endpoint"], "/completions", {
            "prompt": ids, "temperature": 0.0 if role == "attacker" else 1.0,
            "top_p": 1.0, "max_tokens": self.rollout_config.response_length,
            "skip_special_tokens": False,
        })
        choice = result["choices"][0]
        raw = choice["text"].strip()
        # Strip only termination markers; keep the model's reasoning markers.
        for marker in (self.tokenizer.eos_token, self.tokenizer.pad_token):
            if marker and raw.endswith(marker):
                raw = raw[:-len(marker)].rstrip()
        opening = "[THINK]" if self.family == "ministral" else "<think>"
        if self.tokenizer.decode(ids, skip_special_tokens=False).rstrip().endswith(opening) and not raw.startswith(opening):
            raw = opening + raw
        return raw, choice["finish_reason"] == "stop"

    async def _active(self, role, text, sampling, priority):
        ids = self._prompt(role, text)
        output = await self.server_manager.generate(
            request_id=uuid4().hex, prompt_ids=ids, sampling_params=sampling, priority=priority,
        )
        if output.stop_reason == "aborted" or not output.token_ids:
            raise RuntimeError("Active rollout was aborted or returned no tokens")
        finished = (len(output.token_ids) < self.rollout_config.response_length
                    or output.token_ids[-1] == self.tokenizer.eos_token_id)
        return ids, output, generated_text(self.tokenizer, ids, output.token_ids, self.family), finished

    async def _score(self, attack, answer):
        guard = self.settings["guard"]
        generation = {"temperature": 0.0, "top_p": 1.0, "max_tokens": guard.get("max_tokens", 256)}
        query, response = await asyncio.gather(
            request(guard, "/chat/completions", {"messages": [{"role": "user", "content": attack}], **generation}),
            request(guard, "/chat/completions", {"messages": [{"role": "user", "content": attack}, {"role": "assistant", "content": answer}], **generation}),
        )
        for result in (query, response):
            if result["choices"][0]["finish_reason"] != "stop":
                raise ValueError("Guard generation did not complete")
        return defender_reward(query["choices"][0]["message"]["content"], response["choices"][0]["message"]["content"],
                               self.settings["harm_reward"], self.settings["refusal_reward"])

    async def run(self, sampling_params, priority=0, **kwargs):
        priority = int(priority)
        template = kwargs["raw_prompt"][0]["content"]
        seed = kwargs["extra_info"]["raw_prompt"]
        metrics = {"generate_sequences": 0.0, "tool_calls": 0.0, "compute_score": 0.0}
        if self.role == "defender":
            with simple_timer("tool_calls", metrics):
                attack_raw, _ = await self._opponent("attacker", template)
            attack_valid = valid_attacker_format(attack_raw, self.family)
            attack = visible_attack(attack_raw, self.family, seed)
            with simple_timer("generate_sequences", metrics):
                ids, output, answer_raw, finished = await self._active("defender", attack, sampling_params, priority)
            answer = visible_answer(answer_raw, self.family)
        else:
            with simple_timer("generate_sequences", metrics):
                ids, output, attack_raw, finished = await self._active("attacker", template, sampling_params, priority)
            attack_valid = valid_attacker_format(attack_raw, self.family)
            attack = visible_attack(attack_raw, self.family, seed)
            with simple_timer("tool_calls", metrics):
                answer_raw, _ = await self._opponent("defender", attack)
            answer = visible_answer(answer_raw, self.family)
        usable = bool(attack.strip() and answer.strip())
        with simple_timer("compute_score", metrics):
            task = await self._score(attack, answer) if usable else 0.0
        if self.role == "attacker":
            task = -task
        format_score = self.settings["format_reward"] * (1 if attack_valid else -1) if self.role == "attacker" else 0.0
        metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else 0
        return AgentLoopOutput(
            prompt_ids=ids, response_ids=output.token_ids, response_mask=[1] * len(output.token_ids),
            response_logprobs=output.log_probs, routed_experts=output.routed_experts,
            reward_score=(task + format_score) if finished else 0.0, num_turns=2, metrics=metrics,
            extra_fields={"magic_task": task, "magic_invalid": not attack_valid, "magic_usable": usable,
                          "magic_finished": finished, "magic_format": format_score},
        )
