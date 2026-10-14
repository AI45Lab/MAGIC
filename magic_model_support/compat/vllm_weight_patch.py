"""Adapt Mistral3 HF actor weights for a text-only vLLM 0.18 rollout.

Transformers exposes Ministral-3 as ``Mistral3ForConditionalGeneration`` even
for text-only training. Its state dict therefore contains an outer ``model.``
wrapper and multimodal tensors. vLLM 0.18's ``--language-model-only`` path
instantiates ``PixtralForConditionalGeneration`` without those multimodal
modules, while its loader only recognizes multimodal keys that do not have the
outer ``model.`` prefix. VERL's colocated transfer otherwise forwards the
tensors verbatim and fails on ``model.vision_tower.*``.

This module is opt-in via ``runtime/ministral/sitecustomize.py``.
It changes only rollout weight-key adaptation; it does not change model values,
sampling, rewards, optimizer settings, or the training schedule.
"""

from __future__ import annotations

import importlib.metadata
import os
import sys
from collections.abc import Iterable, Iterator
from typing import Any


_MULTIMODAL_COMPONENTS = frozenset(
    {
        "vision_tower",
        "vision_encoder",
        "multi_modal_projector",
        "vision_language_adapter",
        "patch_merger",
        "pre_mm_projector_norm",
    }
)


def adapt_pixtral_text_only_weight_name(name: str) -> str | None:
    """Return the text-only Pixtral loader key, or ``None`` for MM tensors."""

    if any(component in _MULTIMODAL_COMPONENTS for component in name.split(".")):
        return None
    if name in {
        "lm_head.weight",
        "model.language_model.lm_head.weight",
        "language_model.lm_head.weight",
    }:
        return "language_model.lm_head.weight"
    if name.startswith("model.language_model."):
        suffix = name.removeprefix("model.language_model.")
        if suffix.startswith("model."):
            return f"language_model.{suffix}"
        return f"language_model.model.{suffix}"
    return name


def iter_pixtral_text_only_weights(
    weights: Iterable[tuple[str, Any]],
) -> Iterator[tuple[str, Any]]:
    seen_names: set[str] = set()
    for name, tensor in weights:
        adapted_name = adapt_pixtral_text_only_weight_name(name)
        if adapted_name is not None:
            if adapted_name in seen_names:
                raise RuntimeError(
                    f"multiple actor weights map to vLLM parameter {adapted_name}; "
                    "refusing an ambiguous weight update"
                )
            seen_names.add(adapted_name)
            yield adapted_name, tensor


def install_vllm_018_pixtral_text_only_patch() -> None:
    """Patch the exact vLLM 0.18 loader for Ministral text-only models."""

    version = importlib.metadata.version("vllm")
    if version != "0.18.0":
        raise RuntimeError(
            "MAGIC Ministral text-only weight patch requires vLLM 0.18.0; "
            f"found {version}"
        )

    from vllm.model_executor.models.pixtral import PixtralForConditionalGeneration

    if getattr(PixtralForConditionalGeneration, "_magic_text_only_sync_patch", False):
        return

    original_load_weights = PixtralForConditionalGeneration.load_weights

    def load_text_only_weights(self: Any, weights: Iterable[tuple[str, Any]]) -> Any:
        return original_load_weights(self, iter_pixtral_text_only_weights(weights))

    PixtralForConditionalGeneration.load_weights = load_text_only_weights
    PixtralForConditionalGeneration._magic_text_only_sync_patch = True
    print(
        f"MAGIC: installed Ministral text-only vLLM loader patch in pid={os.getpid()}",
        file=sys.stderr,
        flush=True,
    )


def entropy_from_logits_by_token_blocks(logits: Any, chunk_size: int = 128) -> Any:
    """Compute exact categorical entropy while bounding the token-vocab workspace.

    VERL 0.10's helper splits the first tensor dimension.  Padded FSDP logits
    have shape ``[micro_batch, sequence, vocab]``, so a micro batch of eight is
    not divided by its default chunk size of 2048.  Flattening only the leading
    token dimensions lets us bound the softmax workspace without changing the
    entropy formula or its result shape.
    """

    if chunk_size <= 0:
        raise ValueError(f"entropy chunk_size must be positive, got {chunk_size}")

    import torch

    output_shape = logits.shape[:-1]
    flat_logits = logits.reshape(-1, logits.shape[-1])
    entropy_blocks = []
    for logits_block in flat_logits.split(chunk_size, dim=0):
        probabilities = torch.nn.functional.softmax(logits_block, dim=-1)
        entropy_blocks.append(
            torch.logsumexp(logits_block, dim=-1)
            - torch.sum(probabilities * logits_block, dim=-1)
        )
    return torch.cat(entropy_blocks, dim=0).reshape(output_shape)


def logprobs_from_padded_logits_by_sequence(
    logits: Any,
    labels: Any,
    sequence_lengths: list[int],
    logprobs_fn: Any,
) -> Any:
    """Select valid padded rows without concatenating their vocab logits.

    Concatenating valid ``[tokens, vocab]`` slices duplicates several GiB for
    Ministral's 131k vocabulary.  Running the same row-wise cross entropy on
    one sequence view at a time keeps the logits in their original padded
    storage and only concatenates scalar log probabilities.
    """

    import torch

    if logits.ndim != 3:
        raise ValueError(f"expected [batch, sequence, vocab] logits, got {logits.shape}")
    if len(sequence_lengths) != logits.shape[0]:
        raise ValueError(
            f"sequence length count {len(sequence_lengths)} != batch {logits.shape[0]}"
        )
    if any(length <= 0 or length > logits.shape[1] for length in sequence_lengths):
        raise ValueError(
            f"sequence lengths must be in [1, {logits.shape[1]}], got {sequence_lengths}"
        )
    if sum(sequence_lengths) != labels.numel():
        raise ValueError(
            f"sequence lengths sum {sum(sequence_lengths)} != labels {labels.numel()}"
        )

    pieces = []
    label_start = 0
    for batch_index, sequence_length in enumerate(sequence_lengths):
        label_end = label_start + sequence_length
        pieces.append(
            logprobs_fn(
                logits=logits[batch_index, :sequence_length],
                labels=labels[label_start:label_end],
            )
        )
        label_start = label_end
    return torch.cat(pieces, dim=0)


def install_verl_010_no_rmpad_entropy_chunk_patch(chunk_size: int = 128) -> None:
    """Make legacy FSDP's padded path use token-axis entropy chunking.

    VERL 0.10.0.dev0 reads ``entropy_from_logits_with_chunking`` only in its
    remove-padding branch.  The padded branch calls the module-level non-chunked
    helper directly, which duplicates the complete ``tokens x vocab`` tensor.
    Redirect that call to a mathematically equivalent helper which chunks the
    flattened token axis rather than the micro-batch axis.
    """

    version = importlib.metadata.version("verl")
    if version != "0.10.0.dev0":
        raise RuntimeError(
            "MAGIC legacy-FSDP entropy patch requires VERL 0.10.0.dev0; "
            f"found {version}"
        )
    if chunk_size <= 0:
        raise ValueError(f"entropy chunk_size must be positive, got {chunk_size}")

    from verl.utils import torch_functional as verl_F

    if getattr(verl_F.entropy_from_logits, "_magic_chunked_entropy_patch", False):
        return

    def entropy_from_logits(logits: Any) -> Any:
        return entropy_from_logits_by_token_blocks(logits, chunk_size=chunk_size)

    entropy_from_logits._magic_chunked_entropy_patch = True
    verl_F.entropy_from_logits = entropy_from_logits
