"""Batch reward correction shared by both modern MAGIC launchers."""

from __future__ import annotations

from typing import Any

import numpy as np

from magic_model_support.rl.contracts import mask_invalid_attacker_task


def correct_invalid_attacker_rewards(output: Any, prompts: Any) -> Any:
    """Correct the complete rollout batch before VERL computes GRPO advantages.

    Worker chunks can split a prompt's rollouts. Only the manager has the full
    uid group, so this must run after its gather/concat, never inside one worker.
    The supplied prompt uids are also the uids used by the trainer for GRPO.
    """

    fields = output.non_tensor_batch
    roles = fields["magic_train_role"]
    invalid = [
        role == "attacker" and bool(flag)
        for role, flag in zip(roles, fields["attacker_format_invalid"], strict=True)
    ]
    if not any(invalid):
        return output

    uids = prompts.non_tensor_batch.get("uid")
    if uids is None:
        raise ValueError("MAGIC group reward correction requires the trainer's uid")
    if len(uids) != len(roles):
        raise ValueError("MAGIC rollout output must preserve the input batch length/order")
    groups = [(role, uid) for role, uid in zip(roles, uids, strict=True)]
    task = mask_invalid_attacker_task(
        fields["magic_raw_task_reward"], invalid,
        fields["magic_task_reward_usable"], groups,
    )

    scores = output.batch["rm_scores"].clone()
    # VERL stores terminal rewards at the last attended response token, which
    # may be EOS. response_mask is a loss mask and need not include every token.
    lengths = output.batch["attention_mask"][:, -scores.shape[1]:].sum(dim=-1)
    for i, is_invalid in enumerate(invalid):
        if not is_invalid:
            continue
        length = int(lengths[i].item())
        if length <= 0:
            raise ValueError("Cannot assign an attacker format penalty without a response token")
        scores[i].zero_()
        scores[i, length - 1] = task[i] + float(fields["format_reward"][i])

    output.batch["rm_scores"] = scores
    fields["magic_task_reward"] = np.asarray(task, dtype=float)
    return output
