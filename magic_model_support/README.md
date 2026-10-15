# Qwen3.6-27B and Ministral model support

[中文说明](README.zh-CN.md) · [Training entry points](WORKFLOWS.md)

This module provides model loading, checkpoint views, runtime compatibility,
and SFT/MAGIC training entry points for Qwen3.6-27B and
Ministral-3-14B-Reasoning-2512. See the workflow guide for the required environment
variables and launch commands.

## Components

- `fsdp_sft.py`: native FSDP SFT loop, model loaders, token alignment, and HF export.
- `rl/agent_loop.py`, `rl/contracts.py`: active/frozen role rollout and rewards.
- `scripts/run_qwen_sft.sh`, `scripts/run_ministral_sft.sh`: prepared-data SFT launchers.
- `scripts/run_qwen_rl.sh`: controller/service-host Qwen RL launcher.
- `scripts/run_ministral_rl.sh`: local GPU-partitioned Ministral RL launcher.
- `scripts/serve_magic_opponent.sh`, `scripts/serve_qwen3guard.sh`: training services.
- `compat/`: version-specific weight, RoPE, and memory compatibility helpers.
- `runtime/`: opt-in Python startup hooks used by the launchers.

The original upstream separated trainer remains available unchanged. The new
entry points use the documented modern VERL runtime and caller-prepared data.
No standalone benchmark evaluation entry point is included.

## Model interfaces

Qwen3.6 uses the `qwen3_5` / `qwen3_5_text` runtime interfaces. Its SFT loader
extracts the language model and preserves literal ChatML assistant targets.
Ministral uses the native Mistral3 wrapper, its tokenizer template, and independent
output-head handling. Prepared Ministral Reasoning SFT targets use
`[THINK]...[/THINK]answer` and the official reasoning system prompt.

The rollout distinguishes native reasoning from visible output for Ministral.
The exact per-trajectory behavior and configurable checkpoint contents are
described in the [training guide](WORKFLOWS.md).

## Qwen checkpoint views

The base text view validates key/shape metadata against a text checkpoint and
links the source shards without copying their tensor payloads:

```bash
python -m magic_model_support.scripts.prepare_qwen35_text_only_view \
  --source /models/Qwen3.6-27B \
  --reference-text-model /models/qwen-attacker/hf_model \
  --output /models/qwen-text-view

python -m magic_model_support.scripts.prepare_qwen35_vllm_service_view \
  --text-model /models/qwen-text-view \
  --multimodal-config-source /models/Qwen3.6-27B \
  --output /models/qwen-service-view
```

The Qwen RL launcher invokes these helpers automatically. Keep linked source
models available throughout the run.

## Ministral checkpoint normalization

The exporter normalizes a single-file merged checkpoint in place, retaining an
existing independent LM head rather than constructing it from input embeddings:

```bash
python -m magic_model_support.compat.normalize_ministral3_merged \
  --model-dir /models/ministral-export
```

The Ministral RL launcher invokes this after merging. It also checks head and
embedding layout before using the export as a frozen opponent.

## Runtime scope

Use a dedicated runtime for vLLM 0.18.0, the specified VERL development revision,
and Transformers 5 model APIs. The launchers apply the necessary backend patches;
they are not compatible with arbitrary future backend versions. The vLLM package's
declared `transformers<5` bound conflicts with the model API requirement, so the
runtime must be provisioned explicitly. See [runtime setup](WORKFLOWS.md#runtime).

The Qwen path includes text-config rendering, weight-prefix adaptation,
FlashAttention multi-axis position handling, and response-only logits. The
Ministral path includes text-only weight synchronization and RoPE/YaRN handling.
Shared helpers compute entropy and padded log probabilities without materializing
unnecessary full-vocabulary intermediates.
