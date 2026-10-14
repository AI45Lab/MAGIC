#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH}"
TRAIN_FILE="${TRAIN_FILE:?Set TRAIN_FILE}"
VALIDATION_FILE="${VALIDATION_FILE:?Set VALIDATION_FILE}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR}"
NPROC_PER_NODE="${NPROC_PER_NODE:?Set NPROC_PER_NODE}"
MODEL_LOADER="${MODEL_LOADER:-qwen3_5_multimodal_text}"
CHAT_TEMPLATE_MODE="${CHAT_TEMPLATE_MODE:-qwen_chatml_literal}"
CHAT_TEMPLATE_KWARGS_JSON="${CHAT_TEMPLATE_KWARGS_JSON:-}"
if [[ -z "${CHAT_TEMPLATE_KWARGS_JSON}" ]]; then
  CHAT_TEMPLATE_KWARGS_JSON='{}'
fi
EXTRA_ARGS=()

if [[ -n "${MAX_TRAIN_STEPS:-}" ]]; then
  EXTRA_ARGS+=(--max-train-steps "$MAX_TRAIN_STEPS")
fi
if [[ "${CPU_OFFLOAD:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--cpu-offload)
fi

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python -m magic_model_support.scripts.validate_sft_data \
  --model-path "$MODEL_PATH" \
  --train-file "$TRAIN_FILE" \
  --validation-file "$VALIDATION_FILE" \
  --cutoff-len "${CUTOFF_LEN:?Set CUTOFF_LEN}" \
  --model-loader "$MODEL_LOADER" \
  --chat-template-mode "$CHAT_TEMPLATE_MODE" \
  --chat-template-kwargs-json "$CHAT_TEMPLATE_KWARGS_JSON" \
  --max-unsharded-weight-gib "${MAX_UNSHARDED_WEIGHT_GIB:-100}"

torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC_PER_NODE" \
  -m magic_model_support.fsdp_sft \
  --model-path "$MODEL_PATH" \
  --train-file "$TRAIN_FILE" \
  --validation-file "$VALIDATION_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --cutoff-len "${CUTOFF_LEN:?Set CUTOFF_LEN}" \
  --epochs "${EPOCHS:-3}" \
  --micro-batch-size "${MICRO_BATCH_SIZE:-1}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:?Set GRADIENT_ACCUMULATION_STEPS}" \
  --learning-rate "${LEARNING_RATE:-1e-5}" \
  --weight-decay "${WEIGHT_DECAY:-0.0}" \
  --warmup-ratio "${WARMUP_RATIO:-0.1}" \
  --save-steps "${SAVE_STEPS:?Set SAVE_STEPS}" \
  --eval-steps "${EVAL_STEPS:?Set EVAL_STEPS}" \
  --log-steps "${LOG_STEPS:-5}" \
  --seed "${SEED:?Set SEED}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --model-loader "$MODEL_LOADER" \
  --chat-template-mode "$CHAT_TEMPLATE_MODE" \
  --chat-template-kwargs-json "$CHAT_TEMPLATE_KWARGS_JSON" \
  --max-unsharded-weight-gib "${MAX_UNSHARDED_WEIGHT_GIB:-100}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT:-1}" \
  "${EXTRA_ARGS[@]}"
