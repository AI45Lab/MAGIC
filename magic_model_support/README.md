# Qwen3.6-27B and Ministral model adapters

[中文说明](README.zh-CN.md)

This module provides model-level compatibility utilities for **Qwen3.6-27B** and
**Ministral-3-14B-Reasoning-2512** in MAGIC: model loading, SFT token alignment,
checkpoint conversion, native reasoning serialization, and targeted runtime
patches. Complete SFT, MAGIC RL, serving, and evaluation commands are available in the
[workflow guide](WORKFLOWS.md).

## Supported components

| Component | Qwen3.6-27B | Ministral-3-14B-Reasoning-2512 |
|---|---|---|
| Model loading | Extract the text model from the multimodal checkpoint | Load the native Mistral3 wrapper and preserve an independent output head |
| SFT template | Literal ChatML with the assistant target preserved | Native tokenizer template with an explicit prefix-alignment check |
| Checkpoint conversion | Text-only and vLLM serving views | Normalize nested parameter names and LM-head aliases |
| Runtime compatibility | vLLM config/weight mapping and multi-axis FlashAttention position IDs | Text-only Pixtral weight transfer and RoPE/YaRN configuration |
| Reasoning representation | Preserve MAGIC's literal angle-tag target | Convert to `[THINK]...[/THINK]answer`; structured system-message serialization and visible-answer parsing |

The Qwen architecture names `qwen3_5`, `qwen3_5_text`, and
`Qwen3_5ForCausalLM` are the runtime interfaces used by the Qwen3.6 adapter.
Compatibility checks use these architecture fields.

## Runtime requirements

The runtime patches target **vLLM 0.18.0**, Transformers **5.x** model APIs, and
VERL revision
[`5ba5f2f51281522150a6c1c87e3f841d1d220042`](https://github.com/volcengine/verl/tree/5ba5f2f51281522150a6c1c87e3f841d1d220042)
(`0.10.0.dev0`). Use the installed modern VERL package when applying these
patches, rather than MAGIC's vendored `src/verl`.

The vLLM 0.18.0 dependency metadata specifies `transformers<5`, while these
adapters need Transformers 5 model APIs. Provision a compatible isolated
runtime explicitly; a normal dependency-resolved install of vLLM alone is
insufficient. The source patches are version-specific, not universal patches
for arbitrary newer releases.

The complete execution environment uses Python 3.12, PyTorch 2.10.0,
Transformers 5.5.3, safetensors, CUDA, and the appropriate model dependencies.

## Model loading and SFT token alignment

```python
from magic_model_support.sft import (
    load_qwen_text_model,
    load_ministral_model,
    tokenize_sft_record,
)

# Select the loader for the model family. local_rank selects the CUDA device.
model, tokenizer, config = load_qwen_text_model("/models/Qwen3.6-27B", local_rank=0)
# model, tokenizer, config = load_ministral_model(
#     "/models/Ministral-3-14B-Reasoning-2512", local_rank=0
# )

record = {
    "system": "You are a helpful assistant.",
    "instruction": "Explain the example.",
    "input": "Example text",
    "output": "<think>Explanation</think><answer>Answer</answer>",
}
prompt_ids, input_ids = tokenize_sft_record(record, tokenizer, model_family="qwen")
labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
```

Use `model_family="ministral"` for its native tokenizer template, with converted
native targets. `template_kwargs` can pass model-specific template options.
If the rendered prompt is not a prefix of the completed conversation, the
adapter raises an error instead of producing a shifted assistant loss mask.
The loaders return models on the requested CUDA device; the caller owns
parallelism, the optimizer, and checkpoint saving.

## Checkpoint and template conversion

Export Qwen language-model weights without requiring another checkpoint:

```bash
python -m magic_model_support.scripts.prepare_qwen35_text_only_view \
  --source /models/Qwen3.6-27B \
  --output /models/qwen-text-view

python -m magic_model_support.scripts.prepare_qwen35_vllm_service_view \
  --text-model /models/qwen-text-view \
  --multimodal-config-source /models/Qwen3.6-27B \
  --output /models/qwen-service-view
```

The text exporter writes renamed language-model tensors to new shards, leaving
the source unchanged. It needs disk space for the exported text weights. The
serving view links those exported files, which must remain available.

Ministral SFT conversion accepts a JSON array with `instruction`, `input`,
`system`, and `output` fields:

```bash
python -m magic_model_support.scripts.prepare_ministral_reasoning_sft \
  --input /data/sft-input.json \
  --output /data/sft-ministral.json \
  --system-prompt-file /models/Ministral-3-14B-Reasoning-2512/SYSTEM_PROMPT.txt
```

The converter preserves the source, converts canonical angle-tag targets to
native reasoning markers, prepends the supplied official system prompt, and
fails on a noncanonical record before writing output. It preserves record count
and order; it does not select samples or define dataset splits.

To normalize a merged Ministral checkpoint, run the following on the exported
model directory. It updates that directory in place and accepts single-file
or sharded safetensors checkpoints:

```bash
python -m magic_model_support.compat.normalize_ministral3_merged \
  --model-dir /models/ministral-export
```

An existing independent LM head is preserved. Missing or conflicting heads
produce an error; embeddings are not silently substituted for the output head.

For APIs using mistral-common, use
`magic_model_support.reasoning.mistral_reasoning_api_messages` to encode an
example reasoning block in a system message. For raw native model output,
`parse_native_reasoning_response` returns `(reasoning, visible_answer, issue)`;
an unfinished reasoning block is not treated as a visible answer. These helpers
do not assign rewards or implement retries.

## Applying runtime compatibility patches

Work on a dedicated installation or a copied package overlay. Resolve the package
paths **before** enabling runtime hooks:

```bash
VERL_DIR=$(python -c 'import importlib.util; print(next(iter(importlib.util.find_spec("verl").submodule_search_locations)))')
VLLM_DIR=$(python -c 'import importlib.util; print(next(iter(importlib.util.find_spec("vllm").submodule_search_locations)))')
PATCH_DIR="$PWD/magic_model_support/compat/patches"

# Padded log-probability and entropy memory handling, shared by both families.
patch --forward --batch "$VERL_DIR/workers/engine/fsdp/transformer_impl.py" \
  < "$PATCH_DIR/verl010_no_rmpad_logprob.patch"
```

For Qwen:

```bash
patch --forward --batch "$VERL_DIR/workers/engine/fsdp/transformer_impl.py" \
  < "$PATCH_DIR/verl010_qwen35_response_logits.patch"
patch --forward --batch "$VERL_DIR/workers/rollout/vllm_rollout/vllm_async_server.py" \
  < "$PATCH_DIR/verl010_qwen35_text_renderer.patch"

export PYTHONPATH="$PWD/magic_model_support/runtime/qwen:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MAGIC_QWEN35_OUTER_CONFIG_PATH=/models/Qwen3.6-27B/config.json
export MAGIC_QWEN35_FLASH_ATTN_POSIDS_PATCH=1
export MAGIC_RESPONSE_ONLY_LOGITS=1
```

The renderer patch prepares the text-only config after GPU assignment. For a
standalone vLLM service loading text weights through the multimodal wrapper,
set `MAGIC_QWEN35_SFT_PREFIX_ADAPTER=1`. The response-only logits patch requires
padded, non-fused execution (`use_remove_padding=false`); its supported outputs
are policy log probabilities and entropy.

For Ministral:

```bash
patch --forward --batch "$VLLM_DIR/model_executor/models/mistral.py" \
  < "$PATCH_DIR/vllm018_mistral_llama4_rope_parameters.patch"
python -m magic_model_support.compat.patch_vllm018_yarn_mscale \
  --target "$VLLM_DIR/model_executor/layers/rotary_embedding/__init__.py"

export PYTHONPATH="$PWD/magic_model_support/runtime/ministral:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MAGIC_MINISTRAL_TEXT_ONLY_SYNC_PATCH=1
```

Use the hook directory for the selected model family. The Ministral hook adapts
text-only weight transfer and padded entropy calculation; the vLLM model must
use its language-model-only path. Apply patches once to matching source. The
YaRN patcher also recognizes an already-applied patch.

## Changes relative to MAGIC's existing model interfaces

The contribution adds model-specific loading/configuration, token serialization,
and checkpoint-key conversion. Ministral accepts a native reasoning representation
instead of requiring literal angle-tag targets at the tokenizer boundary. Qwen
uses a text-only model view and architecture-specific vLLM configuration. The
memory helpers reduce intermediate vocabulary-logit allocations while preserving
the corresponding log-probability and entropy calculations.

MAGIC's existing trainer and reward implementation are unchanged. The additional modern-runtime entry points are described in the
[workflow guide](WORKFLOWS.md); the original launcher retains its original runtime.
