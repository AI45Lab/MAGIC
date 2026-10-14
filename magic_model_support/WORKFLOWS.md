# Training, serving, and evaluation

[中文说明](WORKFLOWS.zh-CN.md)

The commands below support Qwen3.6-27B and Ministral-3-14B-Reasoning-2512.
Run them from the repository root. Edit the JSON examples to supply local model
and data paths, output paths, and GPU resources. Example resource assignments
are configurable; model memory requirements depend on context and batch size.

## Environment

Use Python 3.12 on Linux with CUDA and PyTorch 2.10.0. Install the modern runtime
separately from the original vendored VERL environment:

```bash
python -m pip install vllm==0.18.0
python -m pip install transformers==5.5.3 \
  'verl @ git+https://github.com/volcengine/verl.git@5ba5f2f51281522150a6c1c87e3f841d1d220042' \
  'ray[default]==2.54.1' accelerate httpx pyarrow
python -m pip install flash-attn==2.8.3 --no-build-isolation

python -m magic_model_support.setup_runtime --model-family qwen
# Or: python -m magic_model_support.setup_runtime --model-family ministral
```

Transformers 5.5.3 provides the model APIs required by this path. Installing it
overrides vLLM 0.18.0's declared `transformers<5` bound, so `pip check` will report
that metadata conflict. The compatibility patches target these exact versions;
do not substitute MAGIC's vendored VERL or silently upgrade the backend.
`setup_runtime` modifies the installed backend source in this dedicated environment.
The two patch sets can be applied sequentially when both families share it.

## SFT

For Qwen, first export the language model:

```bash
python -m magic_model_support.scripts.prepare_qwen35_text_only_view \
  --source /models/Qwen3.6-27B --output /models/qwen-text
```

For Ministral Reasoning, convert the chosen SFT file and repeat for any separate
validation file:

```bash
python -m magic_model_support.scripts.prepare_ministral_reasoning_sft \
  --input /data/sft-source.json --output /data/sft-train.json \
  --system-prompt-file /models/Ministral-3-14B-Reasoning-2512/SYSTEM_PROMPT.txt
```

Inputs are JSON arrays of `instruction`, `input`, `system`, and `output` objects.
The converter changes serialization only. Overlong records cause an error in the
SFT runner; adjust `max_length` or provide compatible data explicitly.

Edit `examples/qwen-sft.json` or `examples/ministral-sft.json`, then run:

```bash
torchrun --standalone --nproc-per-node=4 \
  -m magic_model_support.train_sft --config magic_model_support/examples/qwen-sft.json
# Use ministral-sft.json for Ministral.
```

The examples use FSDP full sharding. GPU count, micro-batch size, accumulation,
precision, and checkpoint frequency are standard Hugging Face `training_args`.
A single-GPU invocation can omit the `fsdp` and `fsdp_config` keys and use `python`.
Set `resume_from_checkpoint` to a Trainer checkpoint directory to resume.
The final HF export is `<output_dir>/hf_model`; Trainer checkpoints also preserve
optimizer and scheduler state. SFT learning rate and epoch examples follow the
repository's full-SFT examples and remain caller-configurable.

## MAGIC RL

Supply prepared upstream-compatible parquet, or convert JSONL rows containing
`attacker_prompt` (the full rewriting task) and `seed_prompt` (the original query):

```bash
python -m magic_model_support.prepare_rl_data \
  --input /data/rl-train.jsonl --output /data/train.parquet
```

Convert a caller-selected validation file separately. The converter neither
samples nor splits data. Upstream parquet must contain `prompt`, `data_source`,
`reward_model`, and `extra_info.raw_prompt`.

Run Qwen3Guard on a separate Guard host, or supply an existing OpenAI-compatible
service:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /models/Qwen3Guard-Gen-8B \
  --served-model-name Qwen3Guard-Gen-8B --port 8000
```

Edit `examples/qwen-rl.json` or `examples/ministral-rl.json`. Set `guard.base_url`,
`guard.model`, and an environment variable named by `guard.api_key_env` if needed;
omit `api_key_env` for a service without authentication. Keep Guard resources
separate from the configured active/opponent devices.

```bash
python -m magic_model_support.train_rl --config magic_model_support/examples/qwen-rl.json
# To continue the same run:
python -m magic_model_support.train_rl --config magic_model_support/examples/qwen-rl.json --resume
```

This entry point manages one local node with separate active-policy and frozen-
opponent GPU sets. It starts the frozen opponent service, runs the active VERL
block, exports the updated policy, and switches roles. It terminates only its own
child processes. `state.json` identifies the latest exported models. Exports are
published atomically after conversion; an interrupted export can be resumed
without repeating an already checkpointed block. Existing checkpoints are kept.

The default schedule and algorithm values follow `scripts/rl/separated/grpo_public.sh`:
defender first, 15 updates per block, 300 combined updates, GRPO group size 4,
batch 64, mini-batch 32, learning rate `1e-6`, and no KL term. Reward weights are
harm 1, refusal 0.5, format 1, matching the public script defaults. These fields
can be set explicitly in the JSON configuration. `micro_batch_size_per_gpu` is
required, and `verl_overrides` accepts Hydra overrides for backend execution
settings. Core model paths, GRPO, checkpoint state, and role scheduling are owned
by the wrapper and take precedence over those overrides.

Both roles save/load model, optimizer, and extra state. The modern AgentLoop uses
the public single-exchange prompts and visible rewritten query, with native
reasoning serialization for Ministral. Invalid attacker task rewards use the
public group's mean valid task reward before GRPO normalization; format penalties
remain separate. Unfinished active responses are masked at the reward level as
in the public manager. Missing visible answers have no task score. No output is
resampled. Malformed Guard responses and transport errors stop the block.

This is a modern VERL execution entry point; the original `separated_trainer`
remains available unchanged. Backend patch requirements and native reasoning
formats are described in [model adapters](README.md).

## Serving

Edit the family-specific service example, setting `model_path` to a base model,
an SFT export, or an RL export:

```bash
python -m magic_model_support.serve --config magic_model_support/examples/qwen-serve.json
python -m magic_model_support.serve --config magic_model_support/examples/ministral-serve.json
```

The Qwen service builds its temporary serving view automatically. Ministral uses
`--tokenizer-mode mistral` with its native reasoning parser; `tekken.json` must be
available in the model directory or at an explicit `tokenizer_path`. The training
exporters preserve that file when it exists in the source checkpoint. For raw
completion output without a reasoning parser, use `--raw`; RL uses this mode.
`cuda_visible_devices`, `tensor_parallel_size`, `max_model_len`, and
`gpu_memory_utilization` control service placement and memory.

## Evaluation

The evaluator accepts JSONL rows with a unique `id` and either `prompt` or
OpenAI-format `messages`. An optional `reference` enables exact-string accuracy.
For Ministral, set `system_prompt_file` to the official prompt and
`tokenizer_mode` to `mistral` to serialize its structured reasoning example.

```bash
python -m magic_model_support.evaluate --config magic_model_support/examples/qwen-eval.json
# Use ministral-eval.json for Ministral.
```

Generation parameters are passed from `generation` to the endpoint. Results
include response, separated reasoning, finish reason, and validity. Only a
nonempty answer with finish reason `stop` is valid for scoring. The adjacent
`.metrics.json` contains total/valid/invalid counts, coverage, and exact-match
accuracy over referenced valid answers. Outputs must use a new filename.

For an API judge, add a `judge` object containing `endpoint`, `generation`, and
`prompt_file`. The caller's template can use `{prompt}`, `{response}`, and
`{reference}`. Require the judge to return a JSON object with a finite numeric
`score`; the evaluator records the judgement and averages scores over valid
answers. The caller defines the criterion and scale. Existing benchmark-specific
evaluators may also target the served OpenAI endpoint.
