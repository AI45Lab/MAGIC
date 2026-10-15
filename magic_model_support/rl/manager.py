"""Modern VERL extension point for MAGIC's group-level attacker correction."""

from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.utils.ray_utils import auto_await

from magic_model_support.rl.rewards import correct_invalid_attacker_rewards


class MagicGameAgentLoopManager(AgentLoopManager):
    @auto_await
    async def generate_sequences(self, prompts):
        output = await super().generate_sequences(prompts)
        return correct_invalid_attacker_rewards(output, prompts)
