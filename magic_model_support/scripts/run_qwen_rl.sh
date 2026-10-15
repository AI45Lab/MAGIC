#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CHECKPOINT_SAVE_CONTENTS:-}" ]]; then
  CHECKPOINT_SAVE_CONTENTS='["model"]'
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${DATA_SEED:?Set DATA_SEED}"
: "${ROLLOUT_SEED:?Set ROLLOUT_SEED}"
TRAIN_FILE="${TRAIN_FILE:?Set TRAIN_FILE}"
BASE_MULTIMODAL_MODEL="${BASE_MULTIMODAL_MODEL:?Set BASE_MULTIMODAL_MODEL}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:?Set SFT_CHECKPOINT}"
GUARD_MODEL="${GUARD_MODEL:?Set GUARD_MODEL}"
RUN_ID="${RUN_ID:?Set the same unique RUN_ID on both hosts}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT}"
RESUME_RUN_ROOT="${RESUME_RUN_ROOT:-}"
RESUME_PHASE="${RESUME_PHASE:-}"
RESUME_MAGIC_STEP="${RESUME_MAGIC_STEP:-0}"
TOTAL_MAGIC_STEPS="${TOTAL_MAGIC_STEPS:-300}"
SWITCH_FREQUENCY="${SWITCH_FREQUENCY:-15}"
ACTIVE_GPUS="${ACTIVE_GPUS:?Set ACTIVE_GPUS}"
OPPONENT_TENSOR_PARALLEL_SIZE="${OPPONENT_TENSOR_PARALLEL_SIZE:?Set OPPONENT_TENSOR_PARALLEL_SIZE}"
GUARD_GPU_MEMORY_UTILIZATION="${GUARD_GPU_MEMORY_UTILIZATION:?Set GUARD_GPU_MEMORY_UTILIZATION}"
OPPONENT_GPU_MEMORY_UTILIZATION="${OPPONENT_GPU_MEMORY_UTILIZATION:?Set OPPONENT_GPU_MEMORY_UTILIZATION}"
AGENT_LOOP_CONFIG="${PROJECT_DIR}/magic_model_support/rl/agent_loops.yaml"
VERL_LOGPROB_PATCH="${PROJECT_DIR}/magic_model_support/compat/patches/verl010_no_rmpad_logprob.patch"
VERL_QWEN_RENDERER_PATCH="${PROJECT_DIR}/magic_model_support/compat/patches/verl010_qwen35_text_renderer.patch"
VERL_QWEN_RESPONSE_LOGITS_PATCH="${PROJECT_DIR}/magic_model_support/compat/patches/verl010_qwen35_response_logits.patch"
VERL_OVERSIZED_MICROBATCH_PATCH="${PROJECT_DIR}/magic_model_support/compat/patches/verl010_allow_oversized_singleton_microbatch.patch"
VERL_SNAPSHOT_SOURCE="${VERL_SNAPSHOT_SOURCE:?Set VERL_SNAPSHOT_SOURCE}"
TEXT_VIEW_BUILDER="${PROJECT_DIR}/magic_model_support/scripts/prepare_qwen35_text_only_view.py"
SERVICE_VIEW_BUILDER="${PROJECT_DIR}/magic_model_support/scripts/prepare_qwen35_vllm_service_view.py"
VLLM_QWEN_ADAPTER_DIR="${PROJECT_DIR}/magic_model_support/runtime/qwen"
COORD_DIR="${RUN_ROOT}/coord"
VAL_FILE="${VAL_FILE:?Set VAL_FILE}"

: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES on each host}"
: "${ACTIVE_GPUS:?Set ACTIVE_GPUS}"
: "${OPPONENT_TENSOR_PARALLEL_SIZE:?Set OPPONENT_TENSOR_PARALLEL_SIZE}"
: "${PPO_MICRO_BATCH_SIZE_PER_GPU:?Set PPO_MICRO_BATCH_SIZE_PER_GPU}"
: "${PPO_MAX_TOKEN_LEN_PER_GPU:?Set PPO_MAX_TOKEN_LEN_PER_GPU}"
IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#visible_devices[@]} != ACTIVE_GPUS )); then
  echo "ERROR: CUDA_VISIBLE_DEVICES must contain ACTIVE_GPUS devices" >&2
  exit 2
fi
if [[ $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ') != "${ACTIVE_GPUS}" ]]; then
  echo "ERROR: host must expose ACTIVE_GPUS devices" >&2
  exit 2
fi
if (( TOTAL_MAGIC_STEPS <= 0 || TOTAL_MAGIC_STEPS % (2 * SWITCH_FREQUENCY) != 0 )); then
  echo "ERROR: TOTAL_MAGIC_STEPS must be a positive multiple of 2*SWITCH_FREQUENCY" >&2
  exit 2
fi
if [[ -n "${RESUME_PHASE}" && "${RESUME_PHASE}" != "defender-step15" ]]; then
  echo "ERROR: unsupported RESUME_PHASE=${RESUME_PHASE}" >&2
  exit 2
fi
if (( RESUME_MAGIC_STEP < 0 || RESUME_MAGIC_STEP >= TOTAL_MAGIC_STEPS || RESUME_MAGIC_STEP % SWITCH_FREQUENCY != 0 )); then
  echo "ERROR: RESUME_MAGIC_STEP must be zero or a completed MAGIC role block before TOTAL_MAGIC_STEPS" >&2
  exit 2
fi
if [[ -n "${RESUME_PHASE}" ]] && (( RESUME_MAGIC_STEP != 0 )); then
  echo "ERROR: RESUME_PHASE and RESUME_MAGIC_STEP are mutually exclusive" >&2
  exit 2
fi
if [[ -n "${RESUME_PHASE}" && -z "${RESUME_RUN_ROOT}" ]]; then
  echo "ERROR: RESUME_RUN_ROOT is required when RESUME_PHASE is set" >&2
  exit 2
fi
if (( RESUME_MAGIC_STEP != 0 )) && [[ -z "${RESUME_RUN_ROOT}" ]]; then
  echo "ERROR: RESUME_RUN_ROOT is required when RESUME_MAGIC_STEP is nonzero" >&2
  exit 2
fi
for path in \
  "${TRAIN_FILE}" \
  "${VAL_FILE}" \
  "${BASE_MULTIMODAL_MODEL}/config.json" \
  "${BASE_MULTIMODAL_MODEL}/model.safetensors.index.json" \
  "${SFT_CHECKPOINT}/hf_model/config.json" \
  "${SFT_CHECKPOINT}/hf_model/model.safetensors" \
  "${GUARD_MODEL}/config.json" \
  "${AGENT_LOOP_CONFIG}" \
  "${VERL_LOGPROB_PATCH}" \
  "${VERL_QWEN_RENDERER_PATCH}" \
  "${VERL_QWEN_RESPONSE_LOGITS_PATCH}" \
  "${VERL_OVERSIZED_MICROBATCH_PATCH}" \
  "${VERL_SNAPSHOT_SOURCE}/__init__.py" \
  "${TEXT_VIEW_BUILDER}" \
  "${SERVICE_VIEW_BUILDER}" \
  "${VLLM_QWEN_ADAPTER_DIR}/sitecustomize.py"; do
  if [[ ! -r "${path}" ]]; then
    echo "ERROR: required path is unreadable: ${path}" >&2
    exit 2
  fi
done

mkdir -p "${COORD_DIR}/commands" "${COORD_DIR}/ready" "${RUN_ROOT}/logs"
hostname_value=$(hostname)
if mkdir "${COORD_DIR}/controller.claim" 2>/dev/null; then
  REPLICA_ROLE=controller
  printf '%s\n' "${hostname_value}" >"${COORD_DIR}/controller.claim/hostname"
else
  REPLICA_ROLE=service
  if ! mkdir "${COORD_DIR}/service.claim" 2>/dev/null; then
    echo "ERROR: more than two replicas attempted to join this run" >&2
    exit 2
  fi
  printf '%s\n' "${hostname_value}" >"${COORD_DIR}/service.claim/hostname"
fi
printf '%s\n' "${REPLICA_ROLE}" >"${RUN_ROOT}/logs/role-${hostname_value}.txt"

VERL_OVERLAY_ROOT="/tmp/magic-verl-overlay-${RUN_ID}"
mkdir -p "${VERL_OVERLAY_ROOT}"
cp -a "${VERL_SNAPSHOT_SOURCE}" "${VERL_OVERLAY_ROOT}/verl"
export MAGIC_QWEN35_OUTER_CONFIG_PATH="${BASE_MULTIMODAL_MODEL}/config.json"
export MAGIC_QWEN35_FLASH_ATTN_POSIDS_PATCH=1
export PYTHONPATH="${VLLM_QWEN_ADAPTER_DIR}:${VERL_OVERLAY_ROOT}:${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
VERL_TRANSFORMER_IMPL="${VERL_OVERLAY_ROOT}/verl/workers/engine/fsdp/transformer_impl.py"
VERL_VLLM_ASYNC_SERVER="${VERL_OVERLAY_ROOT}/verl/workers/rollout/vllm_rollout/vllm_async_server.py"
VERL_SEQLEN_BALANCING="${VERL_OVERLAY_ROOT}/verl/utils/seqlen_balancing.py"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_NO_USAGE_STATS=1
export WANDB_MODE=offline
export WANDB_DIR="${RUN_ROOT}/wandb"
export NCCL_DEBUG=WARN
export RAY_DEDUP_LOGS=0
export MAGIC_OPPONENT_MODEL=magic-opponent
export WILDGUARD_MODEL_NAME=Qwen3Guard-Gen-8B
export MAGIC_HTTP_TIMEOUT_S=900
export MAGIC_HTTP_MAX_RETRIES=2
export MAGIC_RESPONSE_ONLY_LOGITS=1
export REWARD_MAX_TOKENS=256
export REWARD_HARM=1.0
export REWARD_REFUSAL=0.5
export FORMAT_REWARD_VALUE="${FORMAT_REWARD_VALUE:?Set FORMAT_REWARD_VALUE}"
export MAGIC_OPPONENT_MAX_RESAMPLES="${OPPONENT_MAX_RESAMPLES:-3}"

python3 -c 'import tensordict, verl; assert verl.__version__.startswith("0.10.0.dev")'
if [[ "${REPLICA_ROLE}" == controller ]]; then
  python3 -c 'from verl.trainer import main_ppo; from verl.experimental.agent_loop import agent_loop; from verl.workers.engine.fsdp import transformer_impl; from verl import model_merger'
fi

wait_for_file() {
  local path=$1
  local timeout=$2
  local waited=0
  until [[ -s "${path}" ]]; do
    if (( waited >= timeout )); then
      echo "ERROR: timed out waiting for ${path}" >&2
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
}

wait_for_health() {
  local url=$1
  local timeout=$2
  local waited=0
  until curl --fail --silent "${url}" >/dev/null 2>&1; do
    if (( waited >= timeout )); then
      echo "ERROR: timed out waiting for ${url}" >&2
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
}

wait_for_process_health() {
  local pid=$1
  local url=$2
  local timeout=$3
  local waited=0
  until curl --fail --silent "${url}" >/dev/null 2>&1; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "ERROR: process ${pid} exited before ${url} became healthy" >&2
      wait "${pid}"
      return $?
    fi
    if (( waited >= timeout )); then
      echo "ERROR: timed out waiting for ${url}" >&2
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
}

wait_for_process_file() {
  local pid=$1
  local path=$2
  local timeout=$3
  local waited=0
  until [[ -s "${path}" ]]; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "ERROR: process ${pid} exited before ${path} was created" >&2
      wait "${pid}"
      return $?
    fi
    if (( waited >= timeout )); then
      echo "ERROR: timed out waiting for ${path}" >&2
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
}

guard_pid=""
opponent_pid=""
service_cleanup() {
  if [[ -n "${opponent_pid}" ]] && kill -0 "${opponent_pid}" 2>/dev/null; then
    kill -TERM "${opponent_pid}" 2>/dev/null || true
    wait "${opponent_pid}" 2>/dev/null || true
  fi
  if [[ -n "${guard_pid}" ]] && kill -0 "${guard_pid}" 2>/dev/null; then
    kill -TERM "${guard_pid}" 2>/dev/null || true
    wait "${guard_pid}" 2>/dev/null || true
  fi
}

run_service_replica() {
  trap service_cleanup EXIT INT TERM
  wait_for_file "${COORD_DIR}/models.ready" 3600
  service_ip=$(hostname -i | awk '{print $1}')
  if [[ -z "${service_ip}" ]]; then
    echo "ERROR: unable to determine service replica IP" >&2
    exit 2
  fi
  printf '%s\n' "${service_ip}" >"${COORD_DIR}/service_ip.tmp"
  mv "${COORD_DIR}/service_ip.tmp" "${COORD_DIR}/service_ip"

  CUDA_VISIBLE_DEVICES=0 \
    PROJECT_DIR="${PROJECT_DIR}" \
    MODEL_PATH="${GUARD_MODEL}" \
    RUN_DIR="${RUN_ROOT}/services/guard" \
    API_PORT=8000 \
    GPU_MEMORY_UTILIZATION="${GUARD_GPU_MEMORY_UTILIZATION}" \
    bash "${PROJECT_DIR}/magic_model_support/scripts/serve_qwen3guard.sh" \
      >"${RUN_ROOT}/logs/guard.log" 2>&1 &
  guard_pid=$!
  wait_for_process_health "${guard_pid}" "http://127.0.0.1:8000/health" 1800
  printf '%s\n' ready >"${COORD_DIR}/ready/guard.tmp"
  mv "${COORD_DIR}/ready/guard.tmp" "${COORD_DIR}/ready/guard"

  local sequence=1
  while true; do
    local command_file
    command_file=$(printf '%s/commands/%04d.tsv' "${COORD_DIR}" "${sequence}")
    wait_for_file "${command_file}" 86400
    local action model_path phase
    IFS=$'\t' read -r action model_path phase <"${command_file}"
    if [[ "${action}" == shutdown ]]; then
      printf '%s\n' stopped >"${COORD_DIR}/ready/shutdown"
      return 0
    fi
    if [[ "${action}" != serve || -z "${model_path}" || -z "${phase}" ]]; then
      echo "ERROR: invalid coordination command: $(cat "${command_file}")" >&2
      exit 2
    fi
    if [[ -n "${opponent_pid}" ]] && kill -0 "${opponent_pid}" 2>/dev/null; then
      kill -TERM "${opponent_pid}"
      wait "${opponent_pid}" || true
    fi
    local service_dir="${RUN_ROOT}/services/opponent-${phase}"
    rm -f "${service_dir}/endpoint.json"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
      MAGIC_QWEN35_SFT_PREFIX_ADAPTER=1 \
      PYTHONPATH="${VLLM_QWEN_ADAPTER_DIR}:${PYTHONPATH}" \
      MODEL_PATH="${model_path}" \
      SERVED_MODEL_NAME=magic-opponent \
      RUN_DIR="${service_dir}" \
      API_PORT=8001 \
      GPU_MEMORY_UTILIZATION="${OPPONENT_GPU_MEMORY_UTILIZATION}" \
      TENSOR_PARALLEL_SIZE="${OPPONENT_TENSOR_PARALLEL_SIZE}" \
      LANGUAGE_MODEL_ONLY=1 \
      HF_OVERRIDES_JSON='{"architectures":["Qwen3_5ForCausalLM"],"text_config":{"architectures":["Qwen3_5ForCausalLM"]}}' \
      bash "${PROJECT_DIR}/magic_model_support/scripts/serve_magic_opponent.sh" \
        >"${RUN_ROOT}/logs/opponent-${phase}.log" 2>&1 &
    opponent_pid=$!
    wait_for_process_file "${opponent_pid}" "${service_dir}/endpoint.json" 2400
    if ! kill -0 "${opponent_pid}" 2>/dev/null; then
      echo "ERROR: opponent service exited after writing endpoint for ${phase}" >&2
      wait "${opponent_pid}"
    fi
    local ready_file
    ready_file=$(printf '%s/ready/opponent-%04d' "${COORD_DIR}" "${sequence}")
    printf '%s\n' "${phase}" >"${ready_file}.tmp"
    mv "${ready_file}.tmp" "${ready_file}"
    sequence=$((sequence + 1))
  done
}

if [[ "${REPLICA_ROLE}" == service ]]; then
  run_service_replica
  exit 0
fi

command_sequence=0
controller_done=0
send_command() {
  local action=$1
  local model_path=${2:-}
  local phase=${3:-}
  command_sequence=$((command_sequence + 1))
  local target temporary
  target=$(printf '%s/commands/%04d.tsv' "${COORD_DIR}" "${command_sequence}")
  temporary="${target}.tmp.${BASHPID}"
  printf '%s\t%s\t%s\n' "${action}" "${model_path}" "${phase}" >"${temporary}"
  mv "${temporary}" "${target}"
}
controller_cleanup() {
  local status=$?
  if (( controller_done == 0 )); then
    send_command shutdown '' controller-exit || true
  fi
  return "${status}"
}
trap controller_cleanup EXIT INT TERM

same_filesystem() {
  local source=$1
  local target_parent=$2
  [[ "$(stat -c '%d' "${source}")" == "$(stat -c '%d' "${target_parent}")" ]]
}

seed_static_model() {
  local source=$1
  local target=$2
  local target_parent
  target_parent=$(dirname "${target}")
  if same_filesystem "${source}" "${target_parent}"; then
    cp -al "${source}" "${target}"
  else
    ln -s "${source}" "${target}"
  fi
}

seed_resume_checkpoint() {
  local source=$1
  local target=$2
  local target_parent
  target_parent=$(dirname "${target}")
  if same_filesystem "${source}" "${target_parent}"; then
    cp -al "${source}" "${target}"
  else
    echo "Linking resume checkpoint across filesystems: ${source} -> ${target}"
    cp -as "${source}" "${target}"
  fi
}

SFT_SOURCE_MODEL="${SFT_CHECKPOINT}/hf_model"
mkdir -p "${RUN_ROOT}/models"
SFT_MODEL="${RUN_ROOT}/models/attacker-sft"
BASE_MODEL="${RUN_ROOT}/models/defender-base"
seed_static_model "${SFT_SOURCE_MODEL}" "${SFT_MODEL}"
python3 "${TEXT_VIEW_BUILDER}" \
  --source "${BASE_MULTIMODAL_MODEL}" \
  --output "${BASE_MODEL}" \
  --reference-text-model "${SFT_SOURCE_MODEL}" \
  >"${RUN_ROOT}/logs/text-view-build.log" 2>&1
printf '%s\n' ready >"${COORD_DIR}/models.ready.tmp"
mv "${COORD_DIR}/models.ready.tmp" "${COORD_DIR}/models.ready"

wait_for_file "${COORD_DIR}/service_ip" 3600
wait_for_file "${COORD_DIR}/ready/guard" 2400
SERVICE_IP=$(tr -d '\r\n' <"${COORD_DIR}/service_ip")
export MAGIC_OPPONENT_ENDPOINT="http://${SERVICE_IP}:8001/v1"
export WILDGUARD_API_ENDPOINT="http://${SERVICE_IP}:8000/v1"
wait_for_health "http://${SERVICE_IP}:8000/health" 300

python3 -c 'import verl; assert verl.__version__.startswith("0.10.0.dev")'
if patch --dry-run --forward --batch "${VERL_TRANSFORMER_IMPL}" <"${VERL_LOGPROB_PATCH}" >/dev/null 2>&1; then
  patch --forward --batch "${VERL_TRANSFORMER_IMPL}" <"${VERL_LOGPROB_PATCH}"
else
  patch --dry-run --reverse --batch "${VERL_TRANSFORMER_IMPL}" <"${VERL_LOGPROB_PATCH}"
fi
if patch --dry-run --forward --batch "${VERL_TRANSFORMER_IMPL}" <"${VERL_QWEN_RESPONSE_LOGITS_PATCH}" >/dev/null 2>&1; then
  patch --forward --batch "${VERL_TRANSFORMER_IMPL}" <"${VERL_QWEN_RESPONSE_LOGITS_PATCH}"
else
  patch --dry-run --reverse --batch "${VERL_TRANSFORMER_IMPL}" <"${VERL_QWEN_RESPONSE_LOGITS_PATCH}"
fi
if patch --dry-run --forward --batch "${VERL_VLLM_ASYNC_SERVER}" <"${VERL_QWEN_RENDERER_PATCH}" >/dev/null 2>&1; then
  patch --forward --batch "${VERL_VLLM_ASYNC_SERVER}" <"${VERL_QWEN_RENDERER_PATCH}"
else
  patch --dry-run --reverse --batch "${VERL_VLLM_ASYNC_SERVER}" <"${VERL_QWEN_RENDERER_PATCH}"
fi
if patch --dry-run --forward --batch "${VERL_SEQLEN_BALANCING}" <"${VERL_OVERSIZED_MICROBATCH_PATCH}" >/dev/null 2>&1; then
  patch --forward --batch "${VERL_SEQLEN_BALANCING}" <"${VERL_OVERSIZED_MICROBATCH_PATCH}"
else
  patch --dry-run --reverse --batch "${VERL_SEQLEN_BALANCING}" <"${VERL_OVERSIZED_MICROBATCH_PATCH}"
fi
PYTHONPYCACHEPREFIX=/tmp/magic-verl-pycache python3 -m py_compile \
  "${VERL_TRANSFORMER_IMPL}" \
  "${VERL_VLLM_ASYNC_SERVER}" \
  "${VERL_SEQLEN_BALANCING}"

request_opponent() {
  local text_model_path=$1
  local phase=$2
  local service_view="${RUN_ROOT}/service-views/${phase}"
  mkdir -p "${RUN_ROOT}/service-views"
  python3 "${SERVICE_VIEW_BUILDER}" \
    --text-model "${text_model_path}" \
    --multimodal-config-source "${BASE_MULTIMODAL_MODEL}" \
    --output "${service_view}" \
    >"${RUN_ROOT}/logs/service-view-${phase}.log" 2>&1
  send_command serve "${service_view}" "${phase}"
  local ready_file
  ready_file=$(printf '%s/ready/opponent-%04d' "${COORD_DIR}" "${command_sequence}")
  wait_for_file "${ready_file}" 3000
  wait_for_health "http://${SERVICE_IP}:8001/health" 300
}

run_block() {
  local role=$1
  local active_model=$2
  local output_dir=$3
  local target_role_step=$4
  local magic_step=$5
  local checkpoint_load_contents='["model"]'
  local resume_iteration=""
  if [[ -r "${output_dir}/latest_checkpointed_iteration.txt" ]]; then
    resume_iteration=$(tr -d '\r\n' <"${output_dir}/latest_checkpointed_iteration.txt")
  fi
  if [[ -n "${resume_iteration}" ]] \
    && [[ -r "${output_dir}/global_step_${resume_iteration}/actor/optim_world_size_${ACTIVE_GPUS}_rank_0.pt" ]] \
    && [[ -r "${output_dir}/global_step_${resume_iteration}/actor/extra_state_world_size_${ACTIVE_GPUS}_rank_0.pt" ]]; then
    checkpoint_load_contents='["model","optimizer","extra"]'
  fi
  if [[ -n "${CHECKPOINT_LOAD_CONTENTS:-}" ]]; then
    checkpoint_load_contents="${CHECKPOINT_LOAD_CONTENTS}"
  fi
  echo "Checkpoint policy for ${role}: load=${checkpoint_load_contents}, save=${CHECKPOINT_SAVE_CONTENTS}, keep=1"
  export MAGIC_TRAIN_ROLE="${role}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" ray stop --force >/dev/null 2>&1 || true
  MAGIC_ENTROPY_CHUNK_SIZE=128 \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python3 -m verl.trainer.main_ppo \
    trainer.use_v1=false \
    data.train_files="[${TRAIN_FILE}]" \
    data.val_files="[${VAL_FILE}]" \
    data.prompt_key=prompt \
    data.train_batch_size=64 \
    data.val_batch_size=256 \
    data.max_prompt_length=8192 \
    data.max_response_length=6144 \
    data.return_raw_chat=true \
    data.shuffle=true \
    data.seed="${DATA_SEED:?Set DATA_SEED}" \
    data.filter_overlong_prompts=false \
    data.truncation=error \
    actor_rollout_ref.model.path="${active_model}" \
    actor_rollout_ref.model.trust_remote_code=false \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=false \
    +actor_rollout_ref.model.override_config.use_cache=false \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.actor.checkpoint.save_contents="${CHECKPOINT_SAVE_CONTENTS}" \
    actor_rollout_ref.actor.checkpoint.load_contents="${checkpoint_load_contents}" \
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
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
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
    actor_rollout_ref.rollout.prompt_length=8192 \
    actor_rollout_ref.rollout.response_length=6144 \
    actor_rollout_ref.rollout.max_model_len=14336 \
    actor_rollout_ref.rollout.max_num_batched_tokens=49152 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=true \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
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
    trainer.project_name=magic-rl-qwen36-27b \
    trainer.experiment_name="${RUN_ID}-${role}-to${target_role_step}" \
    trainer.logger="[console,wandb]" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${ACTIVE_GPUS}" \
    trainer.total_epochs=1 \
    trainer.total_training_steps="${target_role_step}" \
    trainer.resume_mode=auto \
    trainer.val_before_train=false \
    trainer.save_freq=15 \
    trainer.test_freq=-1 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.default_local_dir="${output_dir}" \
    2>&1 | tee "${RUN_ROOT}/logs/${role}-magic-step${magic_step}-train.log"
  local train_status=${PIPESTATUS[0]}
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" ray stop --force >/dev/null 2>&1 || true
  return "${train_status}"
}

merge_block() {
  local checkpoint_dir=$1
  local target_dir=$2
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${checkpoint_dir}" \
    --target_dir "${target_dir}"
  if [[ ! -r "${target_dir}/model.safetensors" && ! -r "${target_dir}/model.safetensors.index.json" ]]; then
    echo "ERROR: merged checkpoint contains no safetensors weights: ${target_dir}" >&2
    return 1
  fi
}

prune_merged_role() {
  local role=$1
  local keep_dir=$2
  local candidate
  while IFS= read -r -d '' candidate; do
    if [[ "${candidate}" != "${keep_dir}" ]]; then
      echo "Removing superseded merged model (checkpoints are untouched): ${candidate}"
      rm -rf -- "${candidate}"
    fi
  done < <(find "${RUN_ROOT}/merged" -mindepth 1 -maxdepth 1 -type d -name "${role}-step*" -print0 2>/dev/null)
}

DEFENDER_CKPT_ROOT="${RUN_ROOT}/checkpoints/defender"
ATTACKER_CKPT_ROOT="${RUN_ROOT}/checkpoints/attacker"
CURRENT_DEFENDER_HF="${BASE_MODEL}"
CURRENT_ATTACKER_HF="${SFT_MODEL}"
SKIP_DEFENDER_CYCLE=0
START_CYCLE=1
if [[ "${RESUME_PHASE}" == "defender-step15" ]]; then
  resume_defender_checkpoint="${RESUME_RUN_ROOT}/checkpoints/defender/global_step_15"
  resume_defender_merged="${RESUME_RUN_ROOT}/merged/defender-step15"
  if [[ ! -d "${resume_defender_checkpoint}/actor" ]]; then
    echo "ERROR: missing defender checkpoint for resume: ${resume_defender_checkpoint}" >&2
    exit 2
  fi
  if [[ ! -r "${resume_defender_merged}/config.json" ]]; then
    echo "ERROR: missing merged defender model for resume: ${resume_defender_merged}" >&2
    exit 2
  fi
  mkdir -p "${DEFENDER_CKPT_ROOT}"
  seed_resume_checkpoint "${resume_defender_checkpoint}" "${DEFENDER_CKPT_ROOT}/global_step_15"
  printf '%s\n' 15 >"${DEFENDER_CKPT_ROOT}/latest_checkpointed_iteration.txt"
  CURRENT_DEFENDER_HF="${resume_defender_merged}"
  SKIP_DEFENDER_CYCLE=1
fi
if (( RESUME_MAGIC_STEP != 0 )); then
  completed_role_blocks=$((RESUME_MAGIC_STEP / SWITCH_FREQUENCY))
  completed_cycles=$((completed_role_blocks / 2))
  resume_attacker_role_step=$((completed_cycles * SWITCH_FREQUENCY))
  resume_attacker_magic_step=$((completed_cycles * 2 * SWITCH_FREQUENCY))
  if (( completed_role_blocks % 2 == 0 )); then
    resume_defender_role_step=${resume_attacker_role_step}
    resume_defender_magic_step=$((RESUME_MAGIC_STEP - SWITCH_FREQUENCY))
  else
    resume_defender_role_step=$(((completed_cycles + 1) * SWITCH_FREQUENCY))
    resume_defender_magic_step=${RESUME_MAGIC_STEP}
  fi
  resume_defender_checkpoint="${RESUME_RUN_ROOT}/checkpoints/defender/global_step_${resume_defender_role_step}"
  resume_attacker_checkpoint="${RESUME_RUN_ROOT}/checkpoints/attacker/global_step_${resume_attacker_role_step}"
  resume_defender_merged="${RESUME_RUN_ROOT}/merged/defender-step${resume_defender_magic_step}"
  resume_attacker_merged="${RESUME_RUN_ROOT}/merged/attacker-step${resume_attacker_magic_step}"
  required_resume_paths=(
    "${resume_defender_checkpoint}/actor/model_world_size_${ACTIVE_GPUS}_rank_0.pt"
    "${resume_attacker_checkpoint}/actor/model_world_size_${ACTIVE_GPUS}_rank_0.pt"
    "${resume_attacker_merged}/config.json"
  )
  if (( completed_role_blocks % 2 == 0 )); then
    required_resume_paths+=("${resume_defender_merged}/config.json")
  fi
  for path in "${required_resume_paths[@]}"; do
    if [[ ! -r "${path}" ]]; then
      echo "ERROR: missing completed-block resume artifact: ${path}" >&2
      exit 2
    fi
  done
  mkdir -p "${DEFENDER_CKPT_ROOT}" "${ATTACKER_CKPT_ROOT}"
  seed_resume_checkpoint "${resume_defender_checkpoint}" "${DEFENDER_CKPT_ROOT}/global_step_${resume_defender_role_step}"
  seed_resume_checkpoint "${resume_attacker_checkpoint}" "${ATTACKER_CKPT_ROOT}/global_step_${resume_attacker_role_step}"
  printf '%s\n' "${resume_defender_role_step}" >"${DEFENDER_CKPT_ROOT}/latest_checkpointed_iteration.txt"
  printf '%s\n' "${resume_attacker_role_step}" >"${ATTACKER_CKPT_ROOT}/latest_checkpointed_iteration.txt"
  CURRENT_ATTACKER_HF="${resume_attacker_merged}"
  if (( completed_role_blocks % 2 == 0 )); then
    CURRENT_DEFENDER_HF="${resume_defender_merged}"
    START_CYCLE=$((completed_cycles + 1))
    echo "Resuming after completed MAGIC cycle at step ${RESUME_MAGIC_STEP} from ${RESUME_RUN_ROOT}"
  else
    START_CYCLE=$((completed_cycles + 1))
    SKIP_DEFENDER_CYCLE=${START_CYCLE}
    CURRENT_DEFENDER_HF="${RUN_ROOT}/merged/defender-step${resume_defender_magic_step}"
    echo "Rebuilding the missing merged defender view for completed MAGIC step ${RESUME_MAGIC_STEP}"
    merge_block \
      "${DEFENDER_CKPT_ROOT}/global_step_${resume_defender_role_step}/actor" \
      "${CURRENT_DEFENDER_HF}"
    prune_merged_role defender "${CURRENT_DEFENDER_HF}"
    echo "Resuming after completed defender block at MAGIC step ${RESUME_MAGIC_STEP} from ${RESUME_RUN_ROOT}"
  fi
fi
CYCLES=$((TOTAL_MAGIC_STEPS / (2 * SWITCH_FREQUENCY)))

for ((cycle = START_CYCLE; cycle <= CYCLES; cycle++)); do
  role_step=$((cycle * SWITCH_FREQUENCY))
  defender_magic_step=$(((cycle - 1) * 2 * SWITCH_FREQUENCY + SWITCH_FREQUENCY))
  attacker_magic_step=$((cycle * 2 * SWITCH_FREQUENCY))

  if (( cycle == SKIP_DEFENDER_CYCLE )); then
    echo "Skipping already completed defender block for cycle ${cycle} from ${RESUME_RUN_ROOT}"
  else
    request_opponent "${CURRENT_ATTACKER_HF}" "defender-step${defender_magic_step}"
    next_defender_hf="${RUN_ROOT}/merged/defender-step${defender_magic_step}"
    run_block defender "${BASE_MODEL}" "${DEFENDER_CKPT_ROOT}" "${role_step}" "${defender_magic_step}"
    merge_block \
      "${DEFENDER_CKPT_ROOT}/global_step_${role_step}/actor" \
      "${next_defender_hf}"
    prune_merged_role defender "${next_defender_hf}"
    CURRENT_DEFENDER_HF="${next_defender_hf}"
  fi

  request_opponent "${CURRENT_DEFENDER_HF}" "attacker-step${attacker_magic_step}"
  next_attacker_hf="${RUN_ROOT}/merged/attacker-step${attacker_magic_step}"
  run_block attacker "${SFT_MODEL}" "${ATTACKER_CKPT_ROOT}" "${role_step}" "${attacker_magic_step}"
  merge_block \
    "${ATTACKER_CKPT_ROOT}/global_step_${role_step}/actor" \
    "${next_attacker_hf}"
  prune_merged_role attacker "${next_attacker_hf}"
  CURRENT_ATTACKER_HF="${next_attacker_hf}"
done

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

send_command shutdown '' complete
wait_for_file "${COORD_DIR}/ready/shutdown" 300
controller_done=1
