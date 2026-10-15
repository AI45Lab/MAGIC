"""Exercise the actual rollout logic with generation/Guard services stubbed out.

VERL import stubs keep these tests runnable without installing a GPU runtime.
They model only the boundary types; no reward/parsing implementation is mocked.
"""

import asyncio
import importlib.util
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch


def load_rollout_modules():
    modules = {}
    for name in (
        "verl", "verl.experimental", "verl.experimental.agent_loop",
        "verl.experimental.agent_loop.agent_loop", "verl.utils",
        "verl.utils.profiler", "verl.utils.ray_utils", "verl.workers",
        "verl.workers.rollout", "verl.workers.rollout.replica",
    ):
        modules[name] = ModuleType(name)
        modules[name].__path__ = []

    class Manager:
        async def generate_sequences(self, prompts):
            return self.gathered_output

    def auto_await(function):
        def wrapped(*args, **kwargs):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(function(*args, **kwargs))
            return function(*args, **kwargs)
        return wrapped

    agent = modules["verl.experimental.agent_loop.agent_loop"]
    agent.AgentLoopBase = object
    agent.AgentLoopOutput = SimpleNamespace
    agent.AgentLoopManager = Manager
    modules["verl.utils.profiler"].simple_timer = lambda *args: nullcontext()
    modules["verl.utils.ray_utils"].auto_await = auto_await
    modules["verl.workers.rollout.replica"].TokenOutput = SimpleNamespace
    loaded = []
    with patch.dict(sys.modules, modules):
        for name in ("agent_loop", "manager"):
            path = Path(__file__).parents[1] / "rl" / f"{name}.py"
            spec = importlib.util.spec_from_file_location(f"magic_test_{name}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            loaded.append(module)
    return loaded


agent_module, manager_module = load_rollout_modules()


class AgentLoopRewardTests(unittest.IsolatedAsyncioTestCase):
    def make_loop(self, active_text, *, native=False, role="attacker", opponent="defender reply", diagnostic=None):
        loop = object.__new__(agent_module.MagicGameAgentLoop)
        loop.native_reasoning = native
        loop.train_role = role
        loop.format_weight = 0.5
        loop.harm_weight = 1.0
        loop.refusal_weight = 0.5
        loop.response_length = 32
        output = SimpleNamespace(stop_reason="stop", num_preempted=0, extra_fields={}, routed_experts=None)
        loop._active_generate = AsyncMock(return_value=([1], [2, 3], [1, 1], [-0.1, -0.2], active_text, output))
        loop._external_generate = AsyncMock(return_value=(opponent, "stop", diagnostic or {}))
        loop._guard = AsyncMock(return_value=("Safety: Unsafe", "Safety: Safe\nRefusal: Yes"))
        return loop

    async def run_loop(self, loop, *, seed="original question"):
        with patch.dict("os.environ", {"MAGIC_REASONING_SYSTEM_PROMPT_FILE": ""}):
            return await loop.run({}, raw_prompt=[{"role": "user", "content": "rewrite instructions"}],
                                  extra_info={"raw_prompt": seed, "data_type": "harmful"})

    async def test_qwen_only_forwards_answer_and_preserves_trainable_tokens(self):
        loop = self.make_loop("<think>private</think><answer>attack</answer>")
        output = await self.run_loop(loop)
        forwarded = loop._external_generate.call_args.args[0]
        self.assertEqual(forwarded[-1]["content"], "Question:\nattack")
        loop._guard.assert_awaited_once_with("attack", "defender reply")
        self.assertEqual(output.reward_score, -1.0)
        self.assertFalse(output.extra_fields["attacker_format_invalid"])
        self.assertEqual(output.response_ids, [2, 3])
        self.assertEqual(output.response_logprobs, [-0.1, -0.2])

    async def test_malformed_qwen_uses_seed_and_retains_penalty(self):
        for text in ("<think>unfinished", "<think>plan</think><answer></answer>", ""):
            with self.subTest(text=text):
                loop = self.make_loop(text)
                output = await self.run_loop(loop)
                self.assertEqual(loop._external_generate.call_args.args[0][-1]["content"], "Question:\noriginal question")
                self.assertTrue(output.extra_fields["attacker_format_invalid"])
                self.assertEqual(output.extra_fields["format_reward"], -0.5)
                self.assertEqual(output.extra_fields["visible_attacker_output"], "original question")

    async def test_invalid_without_seed_or_scored_response_still_pays_format_penalty(self):
        loop = self.make_loop("<think>unfinished")
        output = await self.run_loop(loop, seed="")
        loop._external_generate.assert_not_awaited()
        loop._guard.assert_not_awaited()
        self.assertEqual(output.reward_score, -0.5)
        self.assertFalse(output.extra_fields["magic_task_reward_usable"])

    async def test_missing_frozen_defender_response_does_not_erase_format_penalty(self):
        loop = self.make_loop("<think>unfinished", opponent="")
        output = await self.run_loop(loop)
        self.assertEqual(output.reward_score, -0.5)
        self.assertFalse(output.extra_fields["magic_task_reward_usable"])
        loop._guard.assert_not_awaited()

    async def test_ministral_native_answer_and_invalid_reasoning(self):
        for text, visible, invalid in (
            ("[THINK]private[/THINK]attack", "attack", False),
            ("[THINK]unfinished", "original question", True),
            ("[THINK][/THINK]attack", "original question", True),
        ):
            with self.subTest(text=text):
                loop = self.make_loop(text, native=True)
                output = await self.run_loop(loop)
                self.assertEqual(output.extra_fields["visible_attacker_output"], visible)
                self.assertEqual(output.extra_fields["attacker_format_invalid"], invalid)
                self.assertEqual(loop._external_generate.call_args.args[0][-1]["content"], f"Question:\n{visible}")

    async def test_frozen_attacker_does_not_receive_defender_format_penalty(self):
        for native, opponent, diagnostic in (
            (False, "<think>unfinished", {}),
            (True, "", {"raw_output": "[THINK]unfinished", "reasoning_issue": "missing_close_marker"}),
        ):
            with self.subTest(native=native):
                response = "[THINK]plan[/THINK]response" if native else "response"
                loop = self.make_loop(response, native=native, role="defender", opponent=opponent, diagnostic=diagnostic)
                output = await self.run_loop(loop)
                self.assertEqual(loop._active_generate.call_args.args[0][-1]["content"], "Question:\noriginal question")
                self.assertEqual(output.reward_score, 1.5)
                self.assertEqual(output.extra_fields["format_reward"], 0.0)

    async def test_external_native_generation_retains_raw_text_for_format_validation(self):
        loop = object.__new__(agent_module.MagicGameAgentLoop)
        loop.native_reasoning = True
        loop.response_length = 32
        loop.opponent_max_model_len = 0
        loop.opponent_max_resamples = 0
        loop.opponent_endpoint = "http://unused"
        loop.opponent_model = "test"
        loop.http_timeout_s = 1
        loop.http_retries = 0
        raw = "[THINK]private[/THINK]attack"
        with patch.object(agent_module, "_chat_completion", new=AsyncMock(return_value=(raw, "stop"))):
            visible, _, diagnostic = await loop._external_generate([], temperature=0)
        self.assertEqual(visible, "attack")
        self.assertEqual(diagnostic["raw_output"], raw)


class ManagerWiringTests(unittest.TestCase):
    def test_manager_applies_correction_to_gathered_batch_in_sync_and_async_calls(self):
        manager = object.__new__(manager_module.MagicGameAgentLoopManager)
        prompts, gathered, corrected = object(), object(), object()
        manager.gathered_output = gathered
        with patch.object(manager_module, "correct_invalid_attacker_rewards", return_value=corrected) as correct:
            self.assertIs(manager.generate_sequences(prompts), corrected)

            async def run():
                return await manager.generate_sequences(prompts)

            self.assertIs(asyncio.run(run()), corrected)
            self.assertEqual(correct.call_count, 2)
            correct.assert_called_with(gathered, prompts)


if __name__ == "__main__":
    unittest.main()
