"""Regression cases for the 20260922 attacker reward contract in modern RL."""

import unittest
from types import SimpleNamespace

import numpy as np

from magic_model_support.rl.contracts import (
    format_reward,
    mask_invalid_attacker_task,
    seed_prompt_from_sample,
    valid_attacker_format,
    visible_attack,
)
from magic_model_support.rl.rewards import correct_invalid_attacker_rewards

try:
    import torch
except ImportError:
    torch = None


class FormatRewardTests(unittest.TestCase):
    def test_qwen_requires_nonempty_ordered_unique_blocks(self):
        valid = "<think>plan</think><answer>attack</answer>"
        self.assertEqual(format_reward(valid), 0.5)
        self.assertEqual(format_reward(" \n" + valid + " \n"), 0.5)
        for invalid in (
            "", "no tags", "<think>unfinished",
            "<think>plan</think><answer></answer>",
            "<think>plan</think><answer> \n </answer>",
            "<think></think><answer>attack</answer>",
            "<think> \n </think><answer>attack</answer>",
            "<answer>attack</answer><think>plan</think>",
            valid + valid, valid + "trailing", "prefix" + valid,
            "[THINK]plan[/THINK]attack",
        ):
            with self.subTest(invalid=invalid):
                self.assertEqual(format_reward(invalid), -0.5)

    def test_ministral_preserves_native_format_with_nonempty_parts(self):
        self.assertEqual(format_reward("[THINK]plan[/THINK]attack", native_reasoning=True), 0.5)
        for invalid in (
            "[THINK]plan", "[THINK]plan[/THINK]", "[THINK]plan[/THINK] \n",
            "[THINK][/THINK]attack", "[THINK] \n [/THINK]attack",
            "[THINK]plan[/THINK][THINK]more[/THINK]attack", "direct answer",
            "<think>plan</think><answer>attack</answer>",
        ):
            with self.subTest(invalid=invalid):
                self.assertEqual(format_reward(invalid, native_reasoning=True), -0.5)

    def test_zero_format_weight_does_not_make_invalid_rewrites_valid(self):
        text = "<think>unfinished"
        self.assertEqual(format_reward(text, 0), 0)
        self.assertFalse(valid_attacker_format(text))
        self.assertEqual(visible_attack(text, "seed"), "seed")

    def test_visible_attack_hides_reasoning_and_uses_seed_for_malformed_tags(self):
        for native, valid, invalid in (
            (False, "<think>private plan</think><answer>attack</answer>", "<think>private plan"),
            (True, "[THINK]private plan[/THINK]attack", "[THINK]private plan"),
        ):
            with self.subTest(native=native):
                self.assertEqual(visible_attack(valid, "seed", native_reasoning=native), "attack")
                self.assertEqual(visible_attack(invalid, "seed", native_reasoning=native), "seed")
                self.assertEqual(visible_attack("", "seed", native_reasoning=native), "seed")
                self.assertEqual(visible_attack("direct", "seed", native_reasoning=native), "direct")

    def test_seed_uses_original_question_not_rewrite_instructions(self):
        self.assertEqual(seed_prompt_from_sample({"question": "seed", "raw_prompt": [{"content": "rewrite"}]}), "seed")
        self.assertEqual(seed_prompt_from_sample({"extra_info": {"raw_prompt": "original"}}), "original")
        self.assertEqual(seed_prompt_from_sample({"raw_prompt": [{"content": "rewrite"}]}), "")


class GroupTaskRewardTests(unittest.TestCase):
    def test_invalid_task_is_valid_group_mean(self):
        task = mask_invalid_attacker_task(
            [-1.5, -1.5, 0.0, 7.0], [False, False, False, True], [True] * 4, ["s"] * 4,
        )
        self.assertEqual(task, [-1.5, -1.5, 0.0, -1.0])

    def test_failed_valid_attacks_beat_invalid_after_format_penalty(self):
        task = mask_invalid_attacker_task(
            [-1.5, -1.5, -1.5, 0.0], [False, False, False, True], [True] * 4, ["s"] * 4,
        )
        self.assertEqual([t + f for t, f in zip(task, [0.5, 0.5, 0.5, -0.5])], [-1, -1, -1, -2])

    def test_interleaved_groups_and_unscored_samples(self):
        task = mask_invalid_attacker_task(
            [-1.5, 1.5, 7.0, 0, 0], [False, False, False, True, True],
            [True, True, False, True, True], ["a", "b", "a", "a", "b"],
        )
        self.assertEqual(task, [-1.5, 1.5, 7.0, -1.5, 1.5])

    def test_all_invalid_keeps_only_format_penalty(self):
        task = mask_invalid_attacker_task([7, 9], [True, True], [True, False], ["s", "s"])
        self.assertEqual([t - 0.5 for t in task], [-0.5, -0.5])

    def test_length_mismatch_fails(self):
        with self.assertRaises(ValueError):
            mask_invalid_attacker_task([0], [True], [], ["s"])


@unittest.skipIf(torch is None, "requires PyTorch for terminal-token reward checks")
class BatchRewardTests(unittest.TestCase):
    def make_batch(self, tasks, invalid, groups, *, usable=None, roles=None):
        count = len(tasks)
        roles = roles or ["attacker"] * count
        fmt = [(-0.5 if bad else 0.5) if role == "attacker" else 0.0
               for role, bad in zip(roles, invalid)]
        # Responses have lengths 1/2/3, with right padding. Deliberately mask
        # EOS out of response_mask: terminal placement must use attention_mask.
        scores = torch.zeros(count, 4)
        attention = torch.zeros(count, 6, dtype=torch.long)
        response_mask = torch.zeros(count, 4, dtype=torch.long)
        for i in range(count):
            length = i % 3 + 1
            attention[i, :2 + length] = 1
            response_mask[i, :length - 1] = 1
            scores[i, length - 1] = tasks[i] + fmt[i]
        output = SimpleNamespace(
            batch={"rm_scores": scores, "attention_mask": attention, "response_mask": response_mask},
            non_tensor_batch={
                "magic_train_role": np.asarray(roles),
                "attacker_format_invalid": np.asarray(invalid),
                "magic_raw_task_reward": np.asarray(tasks, dtype=float),
                "magic_task_reward": np.asarray(tasks, dtype=float),
                "magic_task_reward_usable": np.asarray(usable if usable is not None else [True] * count),
                "format_reward": np.asarray(fmt),
            },
        )
        prompts = SimpleNamespace(non_tensor_batch={"uid": np.asarray(groups)})
        return output, prompts

    def test_complete_batch_correction_across_interleaved_worker_groups(self):
        output, prompts = self.make_batch([0, 1.5, -1.5, 0], [True, False, False, True], ["a", "b", "a", "b"])
        before = output.batch["rm_scores"].clone()
        correct_invalid_attacker_rewards(output, prompts)
        torch.testing.assert_close(output.batch["rm_scores"].sum(-1), torch.tensor([-2., 2., -1., 1.]))
        torch.testing.assert_close(output.batch["rm_scores"][[1, 2]], before[[1, 2]])
        np.testing.assert_equal(output.non_tensor_batch["magic_task_reward"], [-1.5, 1.5, -1.5, 1.5])
        for i, length in enumerate([1, 2, 3, 1]):
            self.assertEqual(output.batch["rm_scores"][i, length:].count_nonzero().item(), 0)
        once = output.batch["rm_scores"].clone()
        correct_invalid_attacker_rewards(output, prompts)
        torch.testing.assert_close(output.batch["rm_scores"], once)

    def test_all_invalid_and_unusable_valid_do_not_supply_a_positive_baseline(self):
        output, prompts = self.make_batch([0, 0, 7], [True, True, False], ["a"] * 3, usable=[False, True, False])
        correct_invalid_attacker_rewards(output, prompts)
        torch.testing.assert_close(output.batch["rm_scores"].sum(-1), torch.tensor([-0.5, -0.5, 7.5]))

    def test_defender_is_unchanged_and_does_not_feed_attacker_mean(self):
        output, prompts = self.make_batch([0, 9], [True, False], ["a", "a"], roles=["attacker", "defender"])
        correct_invalid_attacker_rewards(output, prompts)
        torch.testing.assert_close(output.batch["rm_scores"].sum(-1), torch.tensor([-0.5, 9.]))

    def test_missing_uid_fails_instead_of_using_per_worker_or_singleton_groups(self):
        output, prompts = self.make_batch([0], [True], ["a"])
        prompts.non_tensor_batch.clear()
        with self.assertRaisesRegex(ValueError, "uid"):
            correct_invalid_attacker_rewards(output, prompts)

    def test_no_response_token_fails_instead_of_rewarding_padding(self):
        output, prompts = self.make_batch([0], [True], ["a"])
        output.batch["attention_mask"][:, 2:] = 0
        with self.assertRaisesRegex(ValueError, "response token"):
            correct_invalid_attacker_rewards(output, prompts)


if __name__ == "__main__":
    unittest.main()
