"""Runtime-only vLLM adapter for text-only Qwen3.5 SFT checkpoints."""

from __future__ import annotations

import os
import sys
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def install_qwen35_flash_attention_position_ids_patch() -> None:
    """Do not treat Qwen3.5 multi-axis RoPE positions as packed sequences.

    Transformers 5.3's FlashAttention helper assumes position_ids is 2-D.
    Qwen3.5 passes a multi-axis tensor; for batch size one the old predicate can
    flatten it into invalid cu_seqlens and make FlashAttention read out of
    bounds.  Packed-sequence position ids are always 2-D.
    """
    from transformers import modeling_flash_attention_utils as flash_utils

    if getattr(flash_utils, "_magic_qwen35_position_ids_patch", False):
        return

    original_is_packed_sequence = flash_utils._is_packed_sequence

    def patched_is_packed_sequence(position_ids: Any, batch_size: int) -> bool:
        if position_ids is not None and position_ids.dim() > 2:
            return False
        return original_is_packed_sequence(position_ids, batch_size)

    flash_utils._is_packed_sequence = patched_is_packed_sequence
    flash_utils._magic_qwen35_position_ids_patch = True
    print(
        "MAGIC: installed Qwen3.5 FlashAttention multi-axis position_ids guard",
        file=sys.stderr,
        flush=True,
    )


def _adapt_name(name: str) -> str:
    # The multimodal wrapper's inherited mapper subsequently strips the outer
    # `model.language_model.` prefix and inserts its own `model` component,
    # yielding `language_model.model.*`.
    if name.startswith("model.language_model."):
        # VERL's merger already emits wrapper-prefixed names. Keep this path
        # idempotent so merged RL checkpoints are not prefixed a second time.
        return name
    if name.startswith("model."):
        return "model.language_model." + name[len("model.") :]
    # The inherited mapper already maps lm_head.* -> language_model.lm_head.*.
    return name


def install_qwen35_sft_prefix_adapter() -> None:
    """Patch the vLLM wrapper's weight loader for text-only HF names."""
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration

    model_class = Qwen3_5ForConditionalGeneration
    if not getattr(model_class, "_magic_sft_prefix_adapter", False):
        original_load_weights = model_class.load_weights

        def patched_load_weights(
            self: Any, weights: Iterable[tuple[str, Any]]
        ) -> set[str]:
            adapted = ((_adapt_name(name), tensor) for name, tensor in weights)
            return original_load_weights(self, adapted)

        model_class.load_weights = patched_load_weights
        model_class._magic_sft_prefix_adapter = True
        print(
            "MAGIC: installed Qwen3.5 SFT -> multimodal-wrapper weight-prefix adapter",
            file=sys.stderr,
            flush=True,
        )


if os.environ.get("MAGIC_QWEN35_FLASH_ATTN_POSIDS_PATCH") == "1":
    install_qwen35_flash_attention_position_ids_patch()


if os.environ.get("MAGIC_QWEN35_SFT_PREFIX_ADAPTER") == "1":
    install_qwen35_sft_prefix_adapter()


def prepare_qwen35_text_vllm_config(vllm_config: Any) -> None:
    """Wrap a text-only config after Ray assigns the actor GPU, before AsyncLLM."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config

    model_config = vllm_config.model_config
    text_config = model_config.hf_config
    if not isinstance(text_config, Qwen3_5TextConfig):
        return

    outer_config_path = Path(os.environ["MAGIC_QWEN35_OUTER_CONFIG_PATH"])
    outer_config_dict = json.loads(outer_config_path.read_text(encoding="utf-8"))
    outer_config_dict["text_config"] = text_config.to_dict()
    wrapped_config = Qwen3_5Config(**outer_config_dict)

    model_config.hf_config = wrapped_config
    model_config.hf_text_config = wrapped_config.text_config
    if hasattr(model_config, "get_model_arch_config"):
        model_config.model_arch_config = model_config.get_model_arch_config()

    # Set this only inside the already GPU-bound vLLM HTTP actor. Spawned
    # EngineCore workers inherit it and install the same loader patch through
    # sitecustomize, without importing vLLM in the FSDP WorkerDict processes.
    os.environ["MAGIC_QWEN35_SFT_PREFIX_ADAPTER"] = "1"
    install_qwen35_sft_prefix_adapter()

    print(
        "MAGIC: wrapped Qwen3.5 text config for vLLM language-model-only worker",
        file=sys.stderr,
        flush=True,
    )
