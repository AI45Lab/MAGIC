#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MODEL_LOADER=multimodal_causal_lm
export CHAT_TEMPLATE_MODE=tokenizer
export CHAT_TEMPLATE_KWARGS_JSON='{}'

exec bash "${PROJECT_DIR}/magic_model_support/scripts/run_qwen_sft.sh"
