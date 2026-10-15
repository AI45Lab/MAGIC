#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-magic-opponent}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
API_PORT="${API_PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-14336}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-49152}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-1}"
TOKENIZER_MODE="${TOKENIZER_MODE:-}"
REASONING_PARSER="${REASONING_PARSER:-}"
CONFIG_FORMAT="${CONFIG_FORMAT:-}"
LOAD_FORMAT="${LOAD_FORMAT:-}"
PROBE_MAX_TOKENS="${PROBE_MAX_TOKENS:-1024}"
REQUIRE_NATIVE_REASONING_TAGS="${REQUIRE_NATIVE_REASONING_TAGS:-0}"
PRESERVE_SPECIAL_TOKENS="${PRESERVE_SPECIAL_TOKENS:-0}"
if [[ "${HF_OVERRIDES_JSON+x}" != x ]]; then
  HF_OVERRIDES_JSON='{"text_config":{"architectures":["MistralForCausalLM"]}}'
fi

for required in config.json tokenizer_config.json; do
  if [[ ! -r "${MODEL_PATH}/${required}" ]]; then
    echo "ERROR: missing opponent model file: ${MODEL_PATH}/${required}" >&2
    exit 2
  fi
done
if [[ ! -r "${MODEL_PATH}/model.safetensors" && ! -r "${MODEL_PATH}/model.safetensors.index.json" ]]; then
  echo "ERROR: opponent model has neither a single nor sharded safetensors checkpoint: ${MODEL_PATH}" >&2
  exit 2
fi
if ! command -v vllm >/dev/null 2>&1; then
  echo "ERROR: vllm is unavailable in this image." >&2
  exit 2
fi
if [[ "${MAGIC_QWEN35_SFT_PREFIX_ADAPTER:-0}" == 1 ]]; then
  python3 - <<'PY'
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration

if not getattr(Qwen3_5ForConditionalGeneration, "_magic_sft_prefix_adapter", False):
    raise SystemExit("Qwen3.5 SFT weight-prefix adapter was not installed")
print("Qwen3.5 SFT weight-prefix adapter preflight passed", flush=True)
PY
fi

mkdir -p "${RUN_DIR}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export VLLM_NO_USAGE_STATS=1

vllm_args=(
  serve "${MODEL_PATH}"
  --host 0.0.0.0
  --port "${API_PORT}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --dtype bfloat16
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --enable-chunked-prefill
)
if [[ "${LANGUAGE_MODEL_ONLY}" == 1 ]]; then
  vllm_args+=(--language-model-only)
fi
if [[ -n "${HF_OVERRIDES_JSON}" ]]; then
  vllm_args+=(--hf-overrides "${HF_OVERRIDES_JSON}")
fi
if [[ -n "${TOKENIZER_MODE}" ]]; then
  vllm_args+=(--tokenizer-mode "${TOKENIZER_MODE}")
fi
if [[ -n "${REASONING_PARSER}" ]]; then
  vllm_args+=(--reasoning-parser "${REASONING_PARSER}")
fi
if [[ -n "${CONFIG_FORMAT}" ]]; then
  vllm_args+=(--config-format "${CONFIG_FORMAT}")
fi
if [[ -n "${LOAD_FORMAT}" ]]; then
  vllm_args+=(--load-format "${LOAD_FORMAT}")
fi

vllm "${vllm_args[@]}" &
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
    echo "ERROR: opponent vLLM server exited during startup." >&2
    wait "${server_pid}"
    exit $?
  fi
  if (( waited >= 1800 )); then
    echo "ERROR: timed out waiting for opponent vLLM server." >&2
    exit 1
  fi
  sleep 5
  waited=$((waited + 5))
done

export MAGIC_OPPONENT_RUN_DIR="${RUN_DIR}"
export MAGIC_OPPONENT_BASE_URL="http://127.0.0.1:${API_PORT}/v1"
export MAGIC_OPPONENT_MODEL_NAME="${SERVED_MODEL_NAME}"
export MAGIC_OPPONENT_REASONING_PARSER="${REASONING_PARSER}"
export MAGIC_OPPONENT_TOKENIZER_MODE="${TOKENIZER_MODE}"
export MAGIC_OPPONENT_PROBE_MAX_TOKENS="${PROBE_MAX_TOKENS}"
export MAGIC_OPPONENT_REQUIRE_NATIVE_REASONING_TAGS="${REQUIRE_NATIVE_REASONING_TAGS}"
export MAGIC_OPPONENT_PRESERVE_SPECIAL_TOKENS="${PRESERVE_SPECIAL_TOKENS}"
python3 - <<'PY'
import json
import os
import pathlib
import httpx
from openai import OpenAI
from magic_model_support.rl.contracts import mistral_reasoning_api_messages

base_url = os.environ["MAGIC_OPPONENT_BASE_URL"]
model = os.environ["MAGIC_OPPONENT_MODEL_NAME"]
run_dir = pathlib.Path(os.environ["MAGIC_OPPONENT_RUN_DIR"])
client = OpenAI(base_url=base_url, api_key="local-placeholder", timeout=300, max_retries=0)
messages = []
system_prompt_file = os.environ.get("MAGIC_REASONING_SYSTEM_PROMPT_FILE", "").strip()
if system_prompt_file:
    system_prompt = pathlib.Path(system_prompt_file).read_text(encoding="utf-8").strip()
    if not system_prompt:
        raise SystemExit(f"Reasoning system prompt is empty: {system_prompt_file}")
    messages.append({"role": "system", "content": system_prompt})
messages.append(
    {
        "role": "user",
        "content": (
            "Use the numbers 2, 5, 6, and 3 exactly once, together with arithmetic "
            "operations and parentheses, to make 24. Explain your reasoning and then "
            "give the final expression."
        ),
    }
)
if os.environ.get("MAGIC_OPPONENT_TOKENIZER_MODE") == "mistral":
    messages = mistral_reasoning_api_messages(messages)
request = dict(
    model=model,
    messages=messages,
    temperature=0.0,
    top_p=1.0,
    max_tokens=int(os.environ["MAGIC_OPPONENT_PROBE_MAX_TOKENS"]),
)
preserve_special_tokens = os.environ.get("MAGIC_OPPONENT_PRESERVE_SPECIAL_TOKENS") == "1"
use_stream = bool(os.environ.get("MAGIC_OPPONENT_REASONING_PARSER"))
max_probe_attempts = 1 + int(os.environ.get("MAGIC_OPPONENT_MAX_RESAMPLES", "3"))
if max_probe_attempts <= 0:
    raise SystemExit("MAGIC_OPPONENT_MAX_RESAMPLES must be non-negative")


def run_probe_request():
    reasoning_parts = []
    content_parts = []
    finish_reason = None
    if use_stream:
        raw_chunks = []
        with httpx.Client(timeout=300) as raw_client:
            with raw_client.stream(
                "POST",
                f"{base_url}/chat/completions",
                headers={"Authorization": "Bearer local-placeholder"},
                json={
                    **request,
                    "stream": True,
                    **({"skip_special_tokens": False} if preserve_special_tokens else {}),
                },
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    chunk = json.loads(payload)
                    raw_chunks.append(chunk)
                    if not chunk.get("choices"):
                        continue
                    choice = chunk["choices"][0]
                    delta = choice.get("delta") or {}
                    reasoning_delta = delta.get("reasoning") or delta.get("reasoning_content") or ""
                    if reasoning_delta:
                        reasoning_parts.append(reasoning_delta)
                    if delta.get("content"):
                        content_parts.append(delta["content"])
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
        raw_payload = {"stream": True, "chunks": raw_chunks}
    else:
        extra_body = {"skip_special_tokens": False} if preserve_special_tokens else None
        result = client.chat.completions.create(**request, extra_body=extra_body)
        raw_payload = result.model_dump(mode="json")
        choice = result.choices[0]
        content_parts.append(choice.message.content or "")
        reasoning_parts.append(
            getattr(choice.message, "reasoning", None)
            or getattr(choice.message, "reasoning_content", None)
            or ""
        )
        finish_reason = choice.finish_reason
    return raw_payload, "".join(content_parts), "".join(reasoning_parts), finish_reason


last_probe_error = ""
probe_exhausted_without_visible_response = False
for probe_attempt in range(1, max_probe_attempts + 1):
    raw_payload, text, reasoning, finish_reason = run_probe_request()
    native_open_markers = 0
    native_close_markers = 0
    if not text.strip():
        last_probe_error = "empty completion"
        print(
            f"Reasoning opponent probe attempt {probe_attempt}/{max_probe_attempts} "
            f"failed: {last_probe_error}",
            flush=True,
        )
        continue
    if os.environ.get("MAGIC_OPPONENT_REQUIRE_NATIVE_REASONING_TAGS") == "1":
        open_count = text.count("[THINK]")
        close_count = text.count("[/THINK]")
        native_open_markers = open_count
        native_close_markers = close_count
        if open_count != 1 or close_count < 1:
            last_probe_error = f"no usable native [THINK] block: open={open_count}, close={close_count}"
            print(
                f"Reasoning opponent probe attempt {probe_attempt}/{max_probe_attempts} "
                f"failed: {last_probe_error}",
                flush=True,
            )
            continue
        before, remainder = text.split("[THINK]", 1)
        native_reasoning, final_text = remainder.rsplit("[/THINK]", 1)
        if before.strip():
            last_probe_error = "text before native [THINK] block"
            print(
                f"Reasoning opponent probe attempt {probe_attempt}/{max_probe_attempts} "
                f"failed: {last_probe_error}",
                flush=True,
            )
            continue
        if not native_reasoning.strip() or not final_text.strip():
            last_probe_error = "native [THINK] block or final answer is empty"
            print(
                f"Reasoning opponent probe attempt {probe_attempt}/{max_probe_attempts} "
                f"failed: {last_probe_error}",
                flush=True,
            )
            continue
        reasoning = native_reasoning
        text = final_text.strip()
    if os.environ.get("MAGIC_OPPONENT_REASONING_PARSER") and not reasoning.strip():
        last_probe_error = "no parsed reasoning"
        print(
            f"Reasoning opponent probe attempt {probe_attempt}/{max_probe_attempts} "
            f"failed: {last_probe_error}",
            flush=True,
        )
        continue
    break
else:
    # The training contract already retries an unfinished reasoning trajectory
    # and assigns raw reward 0 when no visible response is produced.  Keep the
    # startup gate consistent with that contract: an unfinished but non-empty
    # native reasoning block proves that the reasoning model is loaded, while
    # the absent visible response is recorded instead of aborting the service.
    if (
        os.environ.get("MAGIC_OPPONENT_REQUIRE_NATIVE_REASONING_TAGS") == "1"
        and native_open_markers == 1
        and native_close_markers == 0
    ):
        _, unfinished_reasoning = text.split("[THINK]", 1)
        if not unfinished_reasoning.strip():
            raise SystemExit(
                f"Reasoning opponent probe failed after {max_probe_attempts} attempts: "
                "native [THINK] block is empty"
            )
        reasoning = unfinished_reasoning
        text = ""
        probe_exhausted_without_visible_response = True
        print(
            f"Reasoning opponent probe exhausted {max_probe_attempts} attempts with "
            "non-empty native reasoning but no visible response; continuing under "
            "the configured reward-0 trajectory policy",
            flush=True,
        )
    else:
        raise SystemExit(
            f"Reasoning opponent probe failed after {max_probe_attempts} attempts: {last_probe_error}"
        )

raw_target = run_dir / "probe.raw.json"
raw_temporary = raw_target.with_suffix(raw_target.suffix + f".tmp.{os.getpid()}")
raw_temporary.write_text(
    json.dumps(raw_payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
raw_temporary.replace(raw_target)
payload = {
    "base_url": base_url,
    "model": model,
    "probe": text,
    "reasoning_chars": len(reasoning),
    "system_prompt_file": system_prompt_file or None,
    "finish_reason": finish_reason,
    "streaming_probe": use_stream,
    "native_reasoning_tags": os.environ.get("MAGIC_OPPONENT_REQUIRE_NATIVE_REASONING_TAGS") == "1",
    "preserve_special_tokens": preserve_special_tokens,
    "native_open_markers": native_open_markers,
    "native_close_markers": native_close_markers,
    "probe_attempt": probe_attempt,
    "probe_max_attempts": max_probe_attempts,
    "probe_visible_response": bool(text.strip()),
    "probe_exhausted_without_visible_response": probe_exhausted_without_visible_response,
}
target = run_dir / "endpoint.json"
temporary = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
temporary.replace(target)
print(f"READY: {payload}", flush=True)
PY

wait "${server_pid}"
