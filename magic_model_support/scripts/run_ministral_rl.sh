#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CHECKPOINT_SAVE_CONTENTS:-}" ]]; then
  CHECKPOINT_SAVE_CONTENTS='["model"]'
fi
CHECKPOINT_LOAD_CONTENTS="${CHECKPOINT_LOAD_CONTENTS:-${CHECKPOINT_SAVE_CONTENTS}}"
export CHECKPOINT_SAVE_CONTENTS

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${DATA_SEED:?Set DATA_SEED}"
: "${ROLLOUT_SEED:?Set ROLLOUT_SEED}"
TRAIN_FILE="${TRAIN_FILE:?Set TRAIN_FILE}"
VAL_FILE="${VAL_FILE:?Set VAL_FILE}"
BASE_SOURCE_MODEL="${BASE_SOURCE_MODEL:?Set BASE_SOURCE_MODEL}"
SFT_ROOT="${SFT_ROOT:-}"
SFT_CHECKPOINT_OVERRIDE="${SFT_CHECKPOINT_OVERRIDE:-}"
SFT_SOURCE_MODEL_OVERRIDE="${SFT_SOURCE_MODEL_OVERRIDE:-}"
GUARD_MODEL="${GUARD_MODEL:?Set GUARD_MODEL}"
RUN_ID="${RUN_ID:?Set a unique RUN_ID}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${RUN_ROOT}}"
AGENT_LOOP_CONFIG="${PROJECT_DIR}/magic_model_support/rl/agent_loops.yaml"
MERGED_NORMALIZER="${PROJECT_DIR}/magic_model_support/compat/normalize_ministral3_merged.py"
VERL_PACKAGE_DIR=$(python3 -c 'import importlib.util; print(next(iter(importlib.util.find_spec("verl").submodule_search_locations)))')
VLLM_PACKAGE_DIR=$(python3 -c 'import importlib.util; print(next(iter(importlib.util.find_spec("vllm").submodule_search_locations)))')
VERL_TRANSFORMER_IMPL="${VERL_PACKAGE_DIR}/workers/engine/fsdp/transformer_impl.py"
VERL_LOGPROB_PATCH="${PROJECT_DIR}/magic_model_support/compat/patches/verl010_no_rmpad_logprob.patch"
VLLM_MISTRAL_IMPL="${VLLM_PACKAGE_DIR}/model_executor/models/mistral.py"
VLLM_LLAMA4_CONFIG_PATCH="${PROJECT_DIR}/magic_model_support/compat/patches/vllm018_mistral_llama4_rope_parameters.patch"
VLLM_ROPE_IMPL="${VLLM_PACKAGE_DIR}/model_executor/layers/rotary_embedding/__init__.py"
VLLM_YARN_MSCALE_PATCHER="${PROJECT_DIR}/magic_model_support/compat/patch_vllm018_yarn_mscale.py"
EXPECTED_CUDA_VISIBLE_DEVICES="${EXPECTED_CUDA_VISIBLE_DEVICES:?Set EXPECTED_CUDA_VISIBLE_DEVICES}"
GUARD_CUDA_VISIBLE_DEVICES="${GUARD_CUDA_VISIBLE_DEVICES:?Set GUARD_CUDA_VISIBLE_DEVICES}"
OPPONENT_CUDA_VISIBLE_DEVICES="${OPPONENT_CUDA_VISIBLE_DEVICES:?Set OPPONENT_CUDA_VISIBLE_DEVICES}"
OPPONENT_TENSOR_PARALLEL_SIZE="${OPPONENT_TENSOR_PARALLEL_SIZE:?Set OPPONENT_TENSOR_PARALLEL_SIZE}"
ACTIVE_CUDA_VISIBLE_DEVICES="${ACTIVE_CUDA_VISIBLE_DEVICES:?Set ACTIVE_CUDA_VISIBLE_DEVICES}"
ACTIVE_GPUS="${ACTIVE_GPUS:?Set ACTIVE_GPUS}"
GUARD_GPU_MEMORY_UTILIZATION="${GUARD_GPU_MEMORY_UTILIZATION:?Set GUARD_GPU_MEMORY_UTILIZATION}"
OPPONENT_GPU_MEMORY_UTILIZATION="${OPPONENT_GPU_MEMORY_UTILIZATION:?Set OPPONENT_GPU_MEMORY_UTILIZATION}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:?Set PPO_MICRO_BATCH_SIZE_PER_GPU}"
TOTAL_MAGIC_STEPS="${TOTAL_MAGIC_STEPS:-300}"
SWITCH_FREQUENCY="${SWITCH_FREQUENCY:-15}"
SAVE_FREQUENCY="${SAVE_FREQUENCY:-15}"
TEST_FREQUENCY="${TEST_FREQUENCY:-15}"
RESUME_AFTER_MAGIC_STEP="${RESUME_AFTER_MAGIC_STEP:-0}"
REASONING_SYSTEM_PROMPT_FILE="${REASONING_SYSTEM_PROMPT_FILE:-${BASE_SOURCE_MODEL}/SYSTEM_PROMPT.txt}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:?Set MAX_PROMPT_LENGTH}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:?Set MAX_RESPONSE_LENGTH}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
OPPONENT_MAX_RESAMPLES="${OPPONENT_MAX_RESAMPLES:?Set OPPONENT_MAX_RESAMPLES}"

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "${EXPECTED_CUDA_VISIBLE_DEVICES}" ]]; then
  echo "ERROR: expected CUDA_VISIBLE_DEVICES=${EXPECTED_CUDA_VISIBLE_DEVICES}, got ${CUDA_VISIBLE_DEVICES:-<unset>}" >&2
  exit 2
fi
if (( TOTAL_MAGIC_STEPS <= 0 || TOTAL_MAGIC_STEPS % (2 * SWITCH_FREQUENCY) != 0 )); then
  echo "ERROR: TOTAL_MAGIC_STEPS must be a positive multiple of 2*SWITCH_FREQUENCY" >&2
  exit 2
fi
if (( RESUME_AFTER_MAGIC_STEP < 0 || RESUME_AFTER_MAGIC_STEP > TOTAL_MAGIC_STEPS || RESUME_AFTER_MAGIC_STEP % SWITCH_FREQUENCY != 0 )); then
  echo "ERROR: RESUME_AFTER_MAGIC_STEP must be a schedule boundary between 0 and TOTAL_MAGIC_STEPS" >&2
  exit 2
fi
if (( PPO_MICRO_BATCH_SIZE_PER_GPU <= 0 )); then
  echo "ERROR: PPO_MICRO_BATCH_SIZE_PER_GPU must be positive" >&2
  exit 2
fi
if (( MAX_MODEL_LEN != MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH )); then
  echo "ERROR: MAX_MODEL_LEN must equal MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH" >&2
  exit 2
fi
for path in "${TRAIN_FILE}" "${VAL_FILE}" "${BASE_SOURCE_MODEL}/config.json" \
  "${GUARD_MODEL}/config.json" "${AGENT_LOOP_CONFIG}" \
  "${MERGED_NORMALIZER}" \
  "${VERL_TRANSFORMER_IMPL}" "${VERL_LOGPROB_PATCH}" \
  "${VLLM_MISTRAL_IMPL}" "${VLLM_LLAMA4_CONFIG_PATCH}" \
  "${VLLM_ROPE_IMPL}" "${VLLM_YARN_MSCALE_PATCHER}" \
  "${REASONING_SYSTEM_PROMPT_FILE}"; do
  if [[ ! -r "${path}" ]]; then
    echo "ERROR: required path is unreadable: ${path}" >&2
    exit 2
  fi
done

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/services" "${ARTIFACT_ROOT}"
if [[ -n "${SFT_CHECKPOINT_OVERRIDE}" ]]; then
  printf '%s\n' "${SFT_CHECKPOINT_OVERRIDE}" > "${RUN_ROOT}/sft_snapshot.txt"
else
  if [[ ! -r "${SFT_ROOT}/latest_checkpoint.txt" ]]; then
    echo "ERROR: missing SFT latest_checkpoint.txt and no override was supplied" >&2
    exit 2
  fi
  cp "${SFT_ROOT}/latest_checkpoint.txt" "${RUN_ROOT}/sft_snapshot.txt"
fi
SFT_CHECKPOINT=$(tr -d '\r\n' < "${RUN_ROOT}/sft_snapshot.txt")
if [[ -n "${SFT_SOURCE_MODEL_OVERRIDE}" ]]; then
  SFT_SOURCE_MODEL="${SFT_SOURCE_MODEL_OVERRIDE}"
else
  SFT_SOURCE_MODEL="${SFT_CHECKPOINT}/hf_model"
fi
if [[ ! -r "${SFT_SOURCE_MODEL}/model.safetensors" ]]; then
  echo "ERROR: snapshotted SFT checkpoint has no HF model: ${SFT_SOURCE_MODEL}" >&2
  exit 2
fi

BASE_MODEL="${BASE_SOURCE_MODEL}"
SFT_MODEL="${SFT_SOURCE_MODEL}"

export PYTHONPATH="${PROJECT_DIR}/magic_model_support/runtime/ministral:${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_NO_USAGE_STATS=1
export WANDB_MODE=offline
export WANDB_DIR="${RUN_ROOT}/wandb"
export NCCL_DEBUG=WARN
export RAY_DEDUP_LOGS=0
export MAGIC_OPPONENT_ENDPOINT="http://127.0.0.1:8001/v1"
export MAGIC_OPPONENT_MODEL="magic-opponent"
export WILDGUARD_API_ENDPOINT="http://127.0.0.1:8000/v1"
export WILDGUARD_MODEL_NAME="Qwen3Guard-Gen-8B"
export MAGIC_HTTP_TIMEOUT_S=900
export MAGIC_HTTP_MAX_RETRIES=2
export MAGIC_OPPONENT_MAX_RESAMPLES="${OPPONENT_MAX_RESAMPLES}"
export REWARD_MAX_TOKENS=256
export REWARD_HARM=1.0
export REWARD_REFUSAL=0.5
export FORMAT_REWARD_VALUE="${FORMAT_REWARD_VALUE:?Set FORMAT_REWARD_VALUE}"
export MAGIC_REASONING_SYSTEM_PROMPT_FILE="${REASONING_SYSTEM_PROMPT_FILE}"
export MAGIC_NATIVE_REASONING=1
export MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH MAX_MODEL_LEN

python3 -c 'import importlib.metadata; assert importlib.metadata.version("verl") == "0.10.0.dev0"'
python3 -c 'import importlib.metadata; assert importlib.metadata.version("vllm") == "0.18.0"'
if patch --dry-run --forward --batch "${VERL_TRANSFORMER_IMPL}" < "${VERL_LOGPROB_PATCH}" >/dev/null 2>&1; then
  patch --forward --batch "${VERL_TRANSFORMER_IMPL}" < "${VERL_LOGPROB_PATCH}"
else
  patch --dry-run --reverse --batch "${VERL_TRANSFORMER_IMPL}" < "${VERL_LOGPROB_PATCH}"
fi
if patch --dry-run --forward --batch "${VLLM_MISTRAL_IMPL}" < "${VLLM_LLAMA4_CONFIG_PATCH}" >/dev/null 2>&1; then
  patch --forward --batch "${VLLM_MISTRAL_IMPL}" < "${VLLM_LLAMA4_CONFIG_PATCH}"
else
  patch --dry-run --reverse --batch "${VLLM_MISTRAL_IMPL}" < "${VLLM_LLAMA4_CONFIG_PATCH}"
fi
python3 "${VLLM_YARN_MSCALE_PATCHER}" --target "${VLLM_ROPE_IMPL}"
PYTHONPYCACHEPREFIX=/tmp/magic-verl-pycache \
  python3 -m py_compile "${VERL_TRANSFORMER_IMPL}"
PYTHONPYCACHEPREFIX=/tmp/magic-vllm-pycache \
  python3 -m py_compile "${VLLM_MISTRAL_IMPL}"
PYTHONPYCACHEPREFIX=/tmp/magic-vllm-pycache \
  python3 -m py_compile "${VLLM_ROPE_IMPL}"
guard_pid=""
opponent_pid=""
cleanup() {
  if [[ -n "${opponent_pid}" ]] && kill -0 "${opponent_pid}" 2>/dev/null; then
    kill -TERM "${opponent_pid}" 2>/dev/null || true
    wait "${opponent_pid}" 2>/dev/null || true
  fi
  if [[ -n "${guard_pid}" ]] && kill -0 "${guard_pid}" 2>/dev/null; then
    kill -TERM "${guard_pid}" 2>/dev/null || true
    wait "${guard_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

start_guard() {
  CUDA_VISIBLE_DEVICES="${GUARD_CUDA_VISIBLE_DEVICES}" \
    PROJECT_DIR="${PROJECT_DIR}" \
    MODEL_PATH="${GUARD_MODEL}" \
    RUN_DIR="${RUN_ROOT}/services/guard" \
    API_PORT=8000 \
    GPU_MEMORY_UTILIZATION="${GUARD_GPU_MEMORY_UTILIZATION}" \
    bash "${PROJECT_DIR}/magic_model_support/scripts/serve_qwen3guard.sh" \
      >"${RUN_ROOT}/logs/guard.log" 2>&1 &
  guard_pid=$!
}

start_opponent() {
  local model_path=$1
  local phase=$2
  local service_view="${RUN_ROOT}/services/model-views/${phase}"
  local source asset
  mkdir -p "${service_view}"
  shopt -s nullglob
  for source in "${model_path}"/*; do
    ln -sfn "${source}" "${service_view}/${source##*/}"
  done
  for asset in tekken.json params.json SYSTEM_PROMPT.txt generation_config.json; do
    if [[ ! -e "${service_view}/${asset}" && -e "${BASE_SOURCE_MODEL}/${asset}" ]]; then
      ln -s "${BASE_SOURCE_MODEL}/${asset}" "${service_view}/${asset}"
    fi
  done
  shopt -u nullglob
  rm -f "${RUN_ROOT}/services/opponent/endpoint.json"
  MAGIC_MINISTRAL_TEXT_ONLY_SYNC_PATCH=1 \
  CUDA_VISIBLE_DEVICES="${OPPONENT_CUDA_VISIBLE_DEVICES}" \
    MODEL_PATH="${service_view}" \
    LANGUAGE_MODEL_ONLY=1 \
    HF_OVERRIDES_JSON='{"text_config":{"architectures":["MistralForCausalLM"]}}' \
    TOKENIZER_MODE='' \
    REASONING_PARSER='' \
    REQUIRE_NATIVE_REASONING_TAGS=1 \
    PRESERVE_SPECIAL_TOKENS=1 \
    PROBE_MAX_TOKENS="${MAX_RESPONSE_LENGTH}" \
    MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    SERVED_MODEL_NAME=magic-opponent \
    RUN_DIR="${RUN_ROOT}/services/opponent" \
    API_PORT=8001 \
    GPU_MEMORY_UTILIZATION="${OPPONENT_GPU_MEMORY_UTILIZATION}" \
    TENSOR_PARALLEL_SIZE="${OPPONENT_TENSOR_PARALLEL_SIZE}" \
    bash "${PROJECT_DIR}/magic_model_support/scripts/serve_magic_opponent.sh" \
      >"${RUN_ROOT}/logs/opponent-${phase}.log" 2>&1 &
  opponent_pid=$!
}

wait_service() {
  local pid=$1
  local health_url=$2
  local label=$3
  local waited=0
  until curl --fail --silent "${health_url}" >/dev/null 2>&1; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "ERROR: ${label} exited before health check passed." >&2
      wait "${pid}"
      return $?
    fi
    if (( waited >= 1800 )); then
      echo "ERROR: timed out waiting for ${label}." >&2
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
}

wait_opponent_ready() {
  local pid=$1
  local label=$2
  local marker="${RUN_ROOT}/services/opponent/endpoint.json"
  local waited=0
  wait_service "${pid}" "http://127.0.0.1:8001/health" "${label}"
  until [[ -s "${marker}" ]]; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "ERROR: ${label} exited before the reasoning probe completed." >&2
      wait "${pid}"
      return $?
    fi
    if (( waited >= 900 )); then
      echo "ERROR: timed out waiting for ${label} reasoning probe marker." >&2
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
  done
  python3 - "${marker}" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if int(payload.get("reasoning_chars", 0)) <= 0:
    raise SystemExit(f"opponent marker has no reasoning output: {payload}")
if not str(payload.get("probe", "")).strip():
    if payload.get("probe_exhausted_without_visible_response") is not True:
        raise SystemExit(f"opponent marker has no final answer: {payload}")
    if payload.get("probe_visible_response") is not False:
        raise SystemExit(f"opponent marker has inconsistent visible-response state: {payload}")
if payload.get("native_reasoning_tags") is not True:
    raise SystemExit(f"opponent marker did not use native reasoning tags: {payload}")
if payload.get("preserve_special_tokens") is not True:
    raise SystemExit(f"opponent marker did not preserve special tokens: {payload}")
print(f"Reasoning opponent gate passed: {payload}", flush=True)
PY
}

stop_opponent() {
  if [[ -n "${opponent_pid}" ]] && kill -0 "${opponent_pid}" 2>/dev/null; then
    kill -TERM "${opponent_pid}"
    wait "${opponent_pid}" || true
  fi
  opponent_pid=""
}

run_block() {
  local role=$1
  local active_model=$2
  local output_dir=$3
  local target_role_step=$4
  local magic_step=$5
  export MAGIC_TRAIN_ROLE="${role}"
  CUDA_VISIBLE_DEVICES="${ACTIVE_CUDA_VISIBLE_DEVICES}" ray stop --force >/dev/null 2>&1 || true
  MAGIC_MINISTRAL_TEXT_ONLY_SYNC_PATCH=1 \
  MAGIC_ENTROPY_CHUNK_SIZE=128 \
  CUDA_VISIBLE_DEVICES="${ACTIVE_CUDA_VISIBLE_DEVICES}" python3 -m verl.trainer.main_ppo \
    trainer.use_v1=false \
    data.train_files="[${TRAIN_FILE}]" \
    data.val_files="[${VAL_FILE}]" \
    data.prompt_key=prompt \
    data.train_batch_size=64 \
    data.val_batch_size=256 \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.return_raw_chat=true \
    data.shuffle=true \
    data.seed="${DATA_SEED:?Set DATA_SEED}" \
    data.filter_overlong_prompts=false \
    data.truncation=error \
    actor_rollout_ref.model.path="${active_model}" \
    actor_rollout_ref.model.trust_remote_code=false \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=false \
    +actor_rollout_ref.model.override_config.tie_word_embeddings=false \
    '+actor_rollout_ref.model.override_config.text_config.architectures=["MistralForCausalLM"]' \
    +actor_rollout_ref.model.override_config.text_config.tie_word_embeddings=false \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.use_dynamic_bsz=false \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=49152 \
    actor_rollout_ref.actor.checkpoint.save_contents="${CHECKPOINT_SAVE_CONTENTS}" \
    actor_rollout_ref.actor.checkpoint.load_contents="${CHECKPOINT_LOAD_CONTENTS}" \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=true \
    actor_rollout_ref.actor.entropy_from_logits_chunk_size=128 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.shuffle=false \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=false \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.data_parallel_size=1 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.seed="${ROLLOUT_SEED:?Set ROLLOUT_SEED}" \
    actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LENGTH}" \
    actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens=49152 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=true \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.text_config.architectures=["MistralForCausalLM"]' \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    actor_rollout_ref.rollout.agent.default_agent_loop=magic_game \
    ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=magic_model_support.rl.manager.MagicGameAgentLoopManager \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=false \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=true \
    algorithm.use_kl_in_reward=false \
    critic.enable=false \
    reward.reward_model.enable=false \
    trainer.project_name=magic-rl-ministral14b \
    trainer.experiment_name="${RUN_ID}-${role}-to${target_role_step}" \
    trainer.logger="[console,wandb]" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${ACTIVE_GPUS}" \
    trainer.total_epochs=1 \
    trainer.total_training_steps="${target_role_step}" \
    trainer.resume_mode=auto \
    trainer.val_before_train=false \
    trainer.save_freq="${SAVE_FREQUENCY}" \
    trainer.test_freq=-1 \
    trainer.max_actor_ckpt_to_keep=20 \
    trainer.default_local_dir="${output_dir}" \
    2>&1 | tee "${RUN_ROOT}/logs/${role}-magic-step${magic_step}-train.log"
  local train_status=${PIPESTATUS[0]}
  CUDA_VISIBLE_DEVICES="${ACTIVE_CUDA_VISIBLE_DEVICES}" ray stop --force >/dev/null 2>&1 || true
  return "${train_status}"
}

merge_block() {
  local checkpoint_dir=$1
  local target_dir=$2
  prepare_untied_checkpoint_config "${checkpoint_dir}"
  CUDA_VISIBLE_DEVICES="${ACTIVE_CUDA_VISIBLE_DEVICES}" python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${checkpoint_dir}" \
    --target_dir "${target_dir}"
  if [[ ! -r "${target_dir}/model.safetensors" && ! -r "${target_dir}/model.safetensors.index.json" ]]; then
    echo "ERROR: merged checkpoint contains no safetensors weights: ${target_dir}" >&2
    return 1
  fi
  normalize_merged "${target_dir}"
  validate_merged_lm_head "${target_dir}"
}

prepare_untied_checkpoint_config() {
  local checkpoint_dir=$1
  python3 - "${checkpoint_dir}/huggingface/config.json" <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
config = json.loads(path.read_text(encoding="utf-8"))
config["tie_word_embeddings"] = False
text_config = config.get("text_config")
if not isinstance(text_config, dict):
    raise RuntimeError(f"missing text_config in {path}")
text_config["tie_word_embeddings"] = False
temporary = path.with_name(f".{path.name}.untied.tmp")
temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, path)
print(json.dumps({"config": str(path), "untied_lm_head": True}), flush=True)
PY
}

validate_merged_lm_head() {
  local target_dir=$1
  python3 - "${target_dir}/model.safetensors" <<'PY'
import json
import sys

import torch
from safetensors import safe_open

path = sys.argv[1]
with safe_open(path, framework="pt", device="cpu") as handle:
    keys = set(handle.keys())
    head_key = "lm_head.weight"
    embed_key = "model.language_model.embed_tokens.weight"
    if head_key not in keys or embed_key not in keys:
        raise RuntimeError(f"merged model is missing untied head/embedding keys: {path}")
    head = handle.get_slice(head_key)
    embed = handle.get_slice(embed_key)
    if head.get_shape() != embed.get_shape():
        raise RuntimeError(
            f"LM head and embeddings have different shapes: {head.get_shape()} vs {embed.get_shape()}"
        )
    rows = head.get_shape()[0]
    sample_rows = sorted({0, 1, rows // 7, rows // 3, rows // 2, rows - 2, rows - 1})
    head_sample = torch.stack([head[row] for row in sample_rows])
    embed_sample = torch.stack([embed[row] for row in sample_rows])
    if torch.equal(head_sample, embed_sample):
        raise RuntimeError(
            "merged LM head is identical to token embeddings on all validation rows; "
            "refusing a tied/cloned checkpoint"
        )
print(json.dumps({
    "model": path,
    "lm_head_key": head_key,
    "embedding_key": embed_key,
    "sample_rows": sample_rows,
    "untied_lm_head": True,
}), flush=True)
PY
}

ensure_resumed_merged() {
  local checkpoint_dir=$1
  local target_dir=$2
  local rebuild_dir="${target_dir}.rebuild.tmp"
  local invalid_dir="${target_dir}.invalid-replaced"

  if [[ -d "${target_dir}" ]] && normalize_merged "${target_dir}" && validate_merged_lm_head "${target_dir}"; then
    return 0
  fi

  rm -rf -- "${rebuild_dir}"
  merge_block "${checkpoint_dir}" "${rebuild_dir}"
  rm -rf -- "${invalid_dir}"
  if [[ -d "${target_dir}" ]]; then
    mv "${target_dir}" "${invalid_dir}"
  fi
  mv "${rebuild_dir}" "${target_dir}"
  validate_merged_lm_head "${target_dir}"
  rm -rf -- "${invalid_dir}"
  echo "Rebuilt resumed merged checkpoint with an independent LM head: ${target_dir}" >&2
}

normalize_merged() {
  local target_dir=$1
  python3 "${MERGED_NORMALIZER}" --model-dir "${target_dir}"
  if [[ ! -r "${target_dir}/model.safetensors" ]]; then
    echo "ERROR: normalized merged model contains no safetensors weights: ${target_dir}" >&2
    return 1
  fi
}

validate_role_checkpoint() {
  local checkpoint_root=$1
  local keep_checkpoint_step=$2
  python3 - "${checkpoint_root}" "${keep_checkpoint_step}" "${ACTIVE_GPUS}" <<'PY'
import json
import pathlib
import sys
import zipfile

checkpoint_root = pathlib.Path(sys.argv[1])
step = int(sys.argv[2])
world_size = int(sys.argv[3])
checkpoint_dir = checkpoint_root / f"global_step_{step}"
actor_dir = checkpoint_dir / "actor"
tracker = checkpoint_root / "latest_checkpointed_iteration.txt"

if not tracker.is_file() or tracker.read_text(encoding="utf-8").strip() != str(step):
    raise RuntimeError(f"checkpoint tracker does not point to step {step}: {tracker}")

shard_sizes = []
for rank in range(world_size):
    shard = actor_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
    if not shard.is_file() or shard.stat().st_size == 0:
        raise RuntimeError(f"missing or undersized model shard: {shard}")
    with zipfile.ZipFile(shard, "r") as archive:
        if not archive.namelist():
            raise RuntimeError(f"empty model shard archive: {shard}")
    shard_sizes.append(shard.stat().st_size)
if len(set(shard_sizes)) != 1:
    raise RuntimeError(f"model shard sizes differ: {shard_sizes}")

data_path = checkpoint_dir / "data.pt"
if not data_path.is_file():
    raise RuntimeError(f"missing dataloader state: {data_path}")
with zipfile.ZipFile(data_path, "r") as archive:
    if not archive.namelist():
        raise RuntimeError(f"empty dataloader state archive: {data_path}")

import os
contents = json.loads(os.environ["CHECKPOINT_SAVE_CONTENTS"])
for component, pattern in (("optimizer", "optim_*.pt"), ("extra", "extra_state_*.pt")):
    if component not in contents:
        forbidden = list(actor_dir.glob(pattern))
        if forbidden:
            raise RuntimeError(f"checkpoint contains unrequested state: {forbidden}")

print(json.dumps({
    "checkpoint_dir": str(checkpoint_dir),
    "step": step,
    "world_size": world_size,
    "shard_bytes": shard_sizes[0],
    "status": "validated",
}), flush=True)
PY
}

prune_role_checkpoints() {
  local checkpoint_root=$1
  local keep_checkpoint_step=$2
  local path base step

  for path in "${checkpoint_root}"/global_step_*; do
    [[ -d "${path}" ]] || continue
    base=${path##*/}
    step=${base#global_step_}
    [[ "${step}" =~ ^[0-9]+$ ]] || continue
    if [[ "${step}" != "${keep_checkpoint_step}" ]]; then
      rm -rf -- "${path}"
    fi
  done
}

prune_role_merged() {
  local merged_root=$1
  local role=$2
  local keep_merged_step=$3
  local path base step
  for path in "${merged_root}/${role}"-step*; do
    [[ -d "${path}" ]] || continue
    base=${path##*/}
    step=${base##*-step}
    [[ "${step}" =~ ^[0-9]+$ ]] || continue
    if [[ "${step}" != "${keep_merged_step}" ]]; then
      rm -rf -- "${path}"
    fi
  done
}

prune_role_artifacts() {
  local checkpoint_root=$1
  local merged_root=$2
  local role=$3
  local keep_checkpoint_step=$4
  local keep_merged_step=$5
  prune_role_checkpoints "${checkpoint_root}" "${keep_checkpoint_step}"
  prune_role_merged "${merged_root}" "${role}" "${keep_merged_step}"
}

retain_latest_attacker() {
  prune_role_artifacts "${ATTACKER_CKPT_ROOT}" "${ARTIFACT_ROOT}/merged" attacker "$1" "$2"
}

start_guard
wait_service "${guard_pid}" "http://127.0.0.1:8000/health" guard
DEFENDER_CKPT_ROOT="${ARTIFACT_ROOT}/checkpoints/defender"
ATTACKER_CKPT_ROOT="${ARTIFACT_ROOT}/checkpoints/attacker"
CURRENT_DEFENDER_HF="${BASE_MODEL}"
CURRENT_ATTACKER_HF="${SFT_MODEL}"
CYCLES=$((TOTAL_MAGIC_STEPS / (2 * SWITCH_FREQUENCY)))
START_CYCLE=$((RESUME_AFTER_MAGIC_STEP / (2 * SWITCH_FREQUENCY) + 1))
COMPLETED_DEFENDER_BLOCKS=$(((RESUME_AFTER_MAGIC_STEP + SWITCH_FREQUENCY) / (2 * SWITCH_FREQUENCY)))
COMPLETED_ATTACKER_BLOCKS=$((RESUME_AFTER_MAGIC_STEP / (2 * SWITCH_FREQUENCY)))

if (( COMPLETED_DEFENDER_BLOCKS > 0 )); then
  latest_defender_magic_step=$(((COMPLETED_DEFENDER_BLOCKS - 1) * 2 * SWITCH_FREQUENCY + SWITCH_FREQUENCY))
  latest_defender_role_step=$((COMPLETED_DEFENDER_BLOCKS * SWITCH_FREQUENCY))
  CURRENT_DEFENDER_HF="${ARTIFACT_ROOT}/merged/defender-step${latest_defender_magic_step}"
  validate_role_checkpoint "${DEFENDER_CKPT_ROOT}" "${latest_defender_role_step}"
  ensure_resumed_merged \
    "${DEFENDER_CKPT_ROOT}/global_step_${latest_defender_role_step}/actor" \
    "${CURRENT_DEFENDER_HF}"
fi
if (( COMPLETED_ATTACKER_BLOCKS > 0 )); then
  latest_attacker_magic_step=$((COMPLETED_ATTACKER_BLOCKS * 2 * SWITCH_FREQUENCY))
  latest_attacker_role_step=$((COMPLETED_ATTACKER_BLOCKS * SWITCH_FREQUENCY))
  CURRENT_ATTACKER_HF="${ARTIFACT_ROOT}/merged/attacker-step${latest_attacker_magic_step}"
  validate_role_checkpoint "${ATTACKER_CKPT_ROOT}" "${latest_attacker_role_step}"
  ensure_resumed_merged \
    "${ATTACKER_CKPT_ROOT}/global_step_${latest_attacker_role_step}/actor" \
    "${CURRENT_ATTACKER_HF}"
  prune_role_artifacts \
    "${ATTACKER_CKPT_ROOT}" \
    "${ARTIFACT_ROOT}/merged" \
    attacker \
    "${latest_attacker_role_step}" \
    "${latest_attacker_magic_step}"
fi

for ((cycle = START_CYCLE; cycle <= CYCLES; cycle++)); do
  role_step=$((cycle * SWITCH_FREQUENCY))
  defender_magic_step=$(((cycle - 1) * 2 * SWITCH_FREQUENCY + SWITCH_FREQUENCY))
  attacker_magic_step=$((cycle * 2 * SWITCH_FREQUENCY))

  next_defender_hf="${ARTIFACT_ROOT}/merged/defender-step${defender_magic_step}"
  if (( RESUME_AFTER_MAGIC_STEP >= defender_magic_step )); then
    if [[ ! -r "${DEFENDER_CKPT_ROOT}/global_step_${role_step}/actor/model_world_size_${ACTIVE_GPUS}_rank_0.pt" ]]; then
      echo "ERROR: missing resumed defender checkpoint for role step ${role_step}" >&2
      exit 2
    fi
    ensure_resumed_merged \
      "${DEFENDER_CKPT_ROOT}/global_step_${role_step}/actor" \
      "${next_defender_hf}"
  else
    stop_opponent
    start_opponent "${CURRENT_ATTACKER_HF}" "defender-step${defender_magic_step}"
    wait_opponent_ready "${opponent_pid}" frozen-attacker
    run_block \
      defender \
      "${BASE_MODEL}" \
      "${DEFENDER_CKPT_ROOT}" \
      "${role_step}" \
      "${defender_magic_step}"
    validate_role_checkpoint "${DEFENDER_CKPT_ROOT}" "${role_step}"
    merge_block \
      "${DEFENDER_CKPT_ROOT}/global_step_${role_step}/actor" \
      "${next_defender_hf}"
  fi
  CURRENT_DEFENDER_HF="${next_defender_hf}"

  stop_opponent
  start_opponent "${CURRENT_DEFENDER_HF}" "attacker-step${attacker_magic_step}"
  wait_opponent_ready "${opponent_pid}" frozen-defender

  next_attacker_hf="${ARTIFACT_ROOT}/merged/attacker-step${attacker_magic_step}"
  if (( RESUME_AFTER_MAGIC_STEP >= attacker_magic_step )); then
    if [[ ! -r "${ATTACKER_CKPT_ROOT}/global_step_${role_step}/actor/model_world_size_${ACTIVE_GPUS}_rank_0.pt" ]]; then
      echo "ERROR: missing resumed attacker checkpoint for role step ${role_step}" >&2
      exit 2
    fi
    ensure_resumed_merged \
      "${ATTACKER_CKPT_ROOT}/global_step_${role_step}/actor" \
      "${next_attacker_hf}"
  else
    run_block \
      attacker \
      "${SFT_MODEL}" \
      "${ATTACKER_CKPT_ROOT}" \
      "${role_step}" \
      "${attacker_magic_step}"
    validate_role_checkpoint "${ATTACKER_CKPT_ROOT}" "${role_step}"
    merge_block \
      "${ATTACKER_CKPT_ROOT}/global_step_${role_step}/actor" \
      "${next_attacker_hf}"
    retain_latest_attacker "${role_step}" "${attacker_magic_step}"
  fi
  CURRENT_ATTACKER_HF="${next_attacker_hf}"
done

stop_opponent

python3 - "${RUN_ROOT}" "${CURRENT_DEFENDER_HF}" "${CURRENT_ATTACKER_HF}" "${TOTAL_MAGIC_STEPS}" <<'PY'
import json
import pathlib
import sys

run_root, defender_hf, attacker_hf, total_magic_steps = sys.argv[1:]
payload = {
    "status": "complete",
    "global_magic_steps": int(total_magic_steps),
    "defender_hf": defender_hf,
    "attacker_hf": attacker_hf,
}
path = pathlib.Path(run_root) / "completion.json"
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2), flush=True)
PY
