"""Apply MAGIC group rewards after gathering all rollout workers."""

import os

import torch
from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.utils.ray_utils import auto_await

from magic_model_support.training.protocol import grouped_attacker_rewards


class MagicAgentLoopManager(AgentLoopManager):
    @auto_await
    async def generate_sequences(self, prompts):
        output = await super().generate_sequences(prompts)
        if os.environ["MAGIC_TRAIN_ROLE"] == "attacker":
            extra = output.non_tensor_batch
            rewards = grouped_attacker_rewards(
                extra["magic_task"], extra["magic_invalid"], extra["magic_usable"],
                extra["magic_finished"], extra["magic_format"], prompts.non_tensor_batch["uid"],
            )
            tensor = output.batch["rm_scores"]
            tensor.zero_()
            last = output.batch["attention_mask"][:, -tensor.shape[1]:].sum(dim=1).long() - 1
            tensor[torch.arange(len(rewards), device=tensor.device), last] = torch.as_tensor(rewards, device=tensor.device)
        return output
