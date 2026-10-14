#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3Guard-Gen-8B}"
RUN_DIR="${RUN_DIR:?Set RUN_DIR}"
API_PORT="${API_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:?Set GPU_MEMORY_UTILIZATION}"
VERL_SNAPSHOT_DIR="${VERL_SNAPSHOT_DIR:-${RUN_DIR}/verl-site-packages-snapshot}"

if [[ ! -r "${MODEL_PATH}/config.json" || ! -r "${MODEL_PATH}/tokenizer_config.json" ]]; then
  echo "ERROR: incomplete Qwen3Guard checkpoint: ${MODEL_PATH}" >&2
  exit 2
fi
if ! command -v vllm >/dev/null 2>&1; then
  echo "ERROR: vllm is not installed in this image." >&2
  exit 2
fi

mkdir -p "${RUN_DIR}"
export GUARD_RUN_DIR="${RUN_DIR}"

# Capture the exact installed VERL package once. This is a read-only source snapshot
# for adapting MAGIC against the image's real API, not an import-path override.
if [[ ! -e "${VERL_SNAPSHOT_DIR}" ]]; then
  verl_package_dir=$(python3 -c 'import pathlib, verl; print(pathlib.Path(verl.__file__).resolve().parent)')
  cp -a "${verl_package_dir}" "${VERL_SNAPSHOT_DIR}"
fi
python3 - <<'PY'
import importlib.metadata
import json
import pathlib
import verl
import os

run_dir = pathlib.Path(os.environ["GUARD_RUN_DIR"])
versions = {
    name: importlib.metadata.version(name)
    for name in ("vllm", "torch", "transformers", "ray")
}
try:
    versions["verl"] = importlib.metadata.version("verl")
except importlib.metadata.PackageNotFoundError:
    versions["verl"] = verl.__version__
versions["verl_import_path"] = str(pathlib.Path(verl.__file__).resolve())
(run_dir / "runtime_versions.json").write_text(
    json.dumps(versions, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export VLLM_NO_USAGE_STATS=1

vllm serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${API_PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enable-chunked-prefill &
server_pid=$!

cleanup() {
  if kill -0 "${server_pid}" 2>/dev/null; then
    kill -TERM "${server_pid}" 2>/dev/null || true
  fi
  wait "${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

waited=0
until curl --fail --silent "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "ERROR: Qwen3Guard vLLM server exited during startup." >&2
    wait "${server_pid}"
    exit $?
  fi
  if (( waited >= 1800 )); then
    echo "ERROR: timed out waiting for Qwen3Guard vLLM server." >&2
    exit 1
  fi
  sleep 5
  waited=$((waited + 5))
done

pod_ip=$(hostname -i | awk '{print $1}')
export GUARD_BASE_URL="http://${pod_ip}:${API_PORT}/v1"
export GUARD_MODEL_NAME="${SERVED_MODEL_NAME}"
python3 - <<'PY'
import json
import os
import pathlib
from openai import OpenAI

base_url = os.environ["GUARD_BASE_URL"]
model = os.environ["GUARD_MODEL_NAME"]
run_dir = pathlib.Path(os.environ["GUARD_RUN_DIR"])
client = OpenAI(base_url=base_url, api_key="local-placeholder", timeout=120, max_retries=0)

prompt_result = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "How do I boil water safely?"}],
    temperature=0.0,
    top_p=1.0,
    max_tokens=128,
)
response_result = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": "How do I boil water safely?"},
        {"role": "assistant", "content": "Use a kettle and avoid touching hot surfaces."},
    ],
    temperature=0.0,
    top_p=1.0,
    max_tokens=128,
)
prompt_text = prompt_result.choices[0].message.content or ""
response_text = response_result.choices[0].message.content or ""
if "Safety:" not in prompt_text:
    raise SystemExit(f"Guard prompt probe lacks Safety label: {prompt_text!r}")
if "Safety:" not in response_text or "Refusal:" not in response_text:
    raise SystemExit(f"Guard response probe lacks Safety/Refusal labels: {response_text!r}")

endpoint = {"base_url": base_url, "model": model}
probe = {"prompt_moderation": prompt_text, "response_moderation": response_text}
for name, value in (("endpoint.json", endpoint), ("probe.json", probe)):
    target = run_dir / name
    temporary = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
print(f"READY: {endpoint}", flush=True)
PY

wait "${server_pid}"
