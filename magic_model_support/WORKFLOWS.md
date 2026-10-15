# Qwen and Ministral training entry points

[中文说明](WORKFLOWS.zh-CN.md)

These entry points use native PyTorch FSDP for SFT and modern VERL for alternating
MAGIC updates. Use dedicated Linux/CUDA hosts: the RL scripts restart the local
Ray processes. Run commands from the repository root and supply absolute paths.
Model and data preparation is performed by the caller. Supply settings through
exported environment variables (`export NAME=value`) or inline environment
assignments on the launch command.

## Runtime

The compatibility layer targets vLLM 0.18.0, PyTorch 2.10, Transformers 5 model
APIs, and VERL revision `5ba5f2f51281522150a6c1c87e3f841d1d220042`
(`0.10.0.dev0`). Use the corresponding isolated runtime, not the vendored
`src/verl` import path. Transformers 5 exceeds vLLM 0.18.0's declared
`transformers<5` bound; this combination requires the provided compatibility
patches rather than relying on dependency resolution alone.

The Qwen launcher applies patches to a copy of `VERL_SNAPSHOT_SOURCE`; set that
variable to an installed VERL package directory or a compatible source snapshot.
The Ministral launcher patches the backend files in its dedicated environment.
Patch reapplication accepts an already-patched source and rejects a mismatched one.

## SFT

Provide prepared JSON arrays with `instruction`, `input`, `system`, and `output`.
Qwen uses literal `<think>...</think><answer>...</answer>` assistant targets.
Ministral uses `[THINK]...[/THINK]answer` and its official reasoning system prompt
in the prepared records. The launchers do not create, split, or select records.

Set these environment variables before launching either family:

| Variable | Meaning |
|---|---|
| `MODEL_PATH` | Original model checkpoint |
| `TRAIN_FILE`, `VALIDATION_FILE` | Prepared SFT JSON arrays |
| `OUTPUT_DIR` | Checkpoint output directory |
| `NPROC_PER_NODE` | Number of local training processes |
| `CUTOFF_LEN` | Tokenization cutoff; longer examples are truncated |
| `GRADIENT_ACCUMULATION_STEPS` | Gradient accumulation count |
| `SAVE_STEPS`, `EVAL_STEPS` | Checkpoint and validation-loss intervals |
| `SEED` | Training seed |

```bash
bash magic_model_support/scripts/run_qwen_sft.sh
# Or:
bash magic_model_support/scripts/run_ministral_sft.sh
```

Optional parameters include `EPOCHS`, `MICRO_BATCH_SIZE`, `LEARNING_RATE`,
`WEIGHT_DECAY`, `WARMUP_RATIO`, `SAVE_TOTAL_LIMIT`, and `CPU_OFFLOAD`.
The output is `checkpoint-<step>/hf_model`, with `latest_checkpoint.txt` identifying
the latest export. The same FSDP forward, loss, optimizer, validation-loss, and
model-export loop is used for both families with their respective model loaders.

## Shared RL inputs

Set `TRAIN_FILE` and `VAL_FILE` to prepared MAGIC-compatible parquet files,
`GUARD_MODEL` to a local Qwen3Guard checkpoint, and `RUN_ROOT` to the run's output
directory. Use a unique `RUN_ID`. No fixed dataset size or benchmark layout is
required by these launchers.

The roles alternate defender first. `TOTAL_MAGIC_STEPS` must be divisible by
`2 * SWITCH_FREQUENCY`. `FORMAT_REWARD_VALUE`, `DATA_SEED`, and `ROLLOUT_SEED`
are supplied explicitly. Backend micro-batches and service memory allocations
must be chosen for the available GPUs.

`CHECKPOINT_SAVE_CONTENTS` and `CHECKPOINT_LOAD_CONTENTS` accept VERL JSON lists,
for example `["model"]` or `["model","optimizer","extra"]`. Choose contents
for the required save/load behavior. Load settings must match the state available
in the checkpoint. When Qwen load contents are not explicitly set, its launcher
detects whether optimizer and extra-state files are available.

## Qwen RL

Run the same script on two dedicated hosts sharing the checkout, checkpoints,
and `RUN_ROOT`. One claims the controller role; the other hosts the frozen
opponent and Guard. The hosts must reach the service ports and shared files.

Required model/runtime variables:

- `BASE_MULTIMODAL_MODEL`: original Qwen checkpoint.
- `SFT_CHECKPOINT`: SFT checkpoint directory containing `hf_model/`.
- `VERL_SNAPSHOT_SOURCE`: compatible VERL package/source directory.
- `CUDA_VISIBLE_DEVICES`, `ACTIVE_GPUS`, `OPPONENT_TENSOR_PARALLEL_SIZE`.
- `PPO_MICRO_BATCH_SIZE_PER_GPU`, `PPO_MAX_TOKEN_LEN_PER_GPU`.
- `GUARD_GPU_MEMORY_UTILIZATION`, `OPPONENT_GPU_MEMORY_UTILIZATION`.

```bash
bash magic_model_support/scripts/run_qwen_rl.sh
```

The controller creates the base text view, requests opponent services through the
shared coordination directory, runs a role block, merges its checkpoint, and
switches the opponent. For a completed-block restart, provide `RESUME_RUN_ROOT`
and `RESUME_MAGIC_STEP` with a fresh `RUN_ROOT`/coordination directory. The launcher
retains the latest merged model per role and uses its configured raw-checkpoint
retention. Source files used by linked model views must remain accessible.

## Ministral RL

Use a dedicated host with separate active-policy and opponent GPU sets. Guard may
share a service GPU if its explicitly configured memory allocation permits it.

Required model/runtime variables:

- `BASE_SOURCE_MODEL`: original Ministral Reasoning checkpoint.
- `SFT_CHECKPOINT_OVERRIDE`: SFT checkpoint containing `hf_model/`; alternatively
  use `SFT_ROOT` with `latest_checkpoint.txt`.
- `REASONING_SYSTEM_PROMPT_FILE`: defaults to the base model's `SYSTEM_PROMPT.txt`.
- `EXPECTED_CUDA_VISIBLE_DEVICES` and matching `CUDA_VISIBLE_DEVICES`.
- `GUARD_CUDA_VISIBLE_DEVICES`, `OPPONENT_CUDA_VISIBLE_DEVICES`,
  `ACTIVE_CUDA_VISIBLE_DEVICES`, `ACTIVE_GPUS`, `OPPONENT_TENSOR_PARALLEL_SIZE`.
- `GUARD_GPU_MEMORY_UTILIZATION`, `OPPONENT_GPU_MEMORY_UTILIZATION`.
- `PPO_MICRO_BATCH_SIZE_PER_GPU`, `MAX_PROMPT_LENGTH`, `MAX_RESPONSE_LENGTH`.
- `OPPONENT_MAX_RESAMPLES`: maximum additional attempts after an opponent output
  with no visible answer.

```bash
bash magic_model_support/scripts/run_ministral_rl.sh
```

The script starts Guard and the frozen opponent, waits for readiness, executes the
active role, merges/normalizes the checkpoint, and switches roles. Set
`RESUME_AFTER_MAGIC_STEP` to a completed role boundary when restarting the same
run. The latest attacker artifacts are retained. Defender artifacts are kept
without benchmark-based selection. The export normalizer expects a single
`model.safetensors` file and preserves the independent LM head.

## Rollout behavior

The active role's tokens and log probabilities come from VERL's rollout engine;
the frozen opponent and Guard use OpenAI-compatible services. Qwen attackers
forward only `<answer>` content; Ministral attackers forward the visible answer
after `[THINK]...[/THINK]`. Both formats require non-empty reasoning and answers,
with complete, unique, ordered markers. Untagged direct answers can still be
forwarded but fail the attacker format check. Malformed tagged or empty attacker
output falls back to the original seed from `seed_prompt`, `question`, or
`extra_info.raw_prompt`. Defender scoring retains its existing response handling.

Native-reasoning opponents are resampled up to the configured limit. When no
response can be scored, the raw task reward is zero, but attacker format rewards
and penalties still apply. Unscored trajectories do not contribute to group means.

Both launchers use `MagicGameAgentLoopManager` to gather all worker outputs before
GRPO normalization. As in the `20260922` trainer, an invalid attacker's task reward
is replaced with the mean of usable, valid attacks with the same training `uid`,
then its format penalty is added. If no usable valid attack exists in the group,
the replacement task reward is zero and the format penalty remains. Custom
launchers must also set
`actor_rollout_ref.rollout.agent.agent_loop_manager_class=magic_model_support.rl.manager.MagicGameAgentLoopManager`.

The scripts `serve_magic_opponent.sh` and `serve_qwen3guard.sh` implement the
service launch/readiness protocol used by the training entry points. This
package does not provide standalone benchmark or API-judge evaluation commands.
