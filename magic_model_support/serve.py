"""Serve a Qwen or Ministral checkpoint through a vLLM OpenAI API."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import signal
import tempfile


def service_command(config, model_path, *, raw=False):
    family = config["model_family"]
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", str(model_path), "--served-model-name", config["served_model_name"],
               "--host", config.get("host", "127.0.0.1"), "--port", str(config["port"]),
               "--tensor-parallel-size", str(config["tensor_parallel_size"]),
               "--dtype", "bfloat16", "--max-model-len", str(config["max_model_len"]),
               "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
               "--language-model-only"]
    if family == "ministral":
        command += ["--config-format", "hf", "--hf-overrides", json.dumps({"text_config": {"architectures": ["MistralForCausalLM"]}})]
        if not raw:
            command += ["--tokenizer-mode", "mistral"]
    elif family != "qwen":
        raise ValueError(f"Unsupported model_family: {family}")
    if not raw:
        command += ["--reasoning-parser", "qwen3" if family == "qwen" else "mistral"]
    if config.get("tokenizer_path"):
        command += ["--tokenizer", config["tokenizer_path"]]
    command += config.get("extra_args", [])
    return command


def runtime_environment(family):
    package = Path(__file__).resolve().parent
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(package / "runtime" / family), str(package.parent), env.get("PYTHONPATH", "")])
    if family == "qwen":
        env.pop("MAGIC_QWEN35_SFT_PREFIX_ADAPTER", None)
        env["MAGIC_QWEN35_FLASH_ATTN_POSIDS_PATCH"] = "1"
    elif family == "ministral":
        env["MAGIC_MINISTRAL_TEXT_ONLY_SYNC_PATCH"] = "1"
    else:
        raise ValueError(f"Unsupported model_family: {family}")
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", type=Path, help="Override the checkpoint in the config")
    parser.add_argument("--raw", action="store_true", help="Return raw tokens without a reasoning parser")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    model = args.model or Path(config["model_path"])
    env = runtime_environment(config["model_family"])
    if config.get("cuda_visible_devices") is not None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config["cuda_visible_devices"]))
    with tempfile.TemporaryDirectory(prefix="magic-service-") as temporary:
        if config["model_family"] == "qwen":
            env["MAGIC_QWEN35_SFT_PREFIX_ADAPTER"] = "1"
            source = config["qwen_config_source"]
            target = Path(temporary) / "model"
            subprocess.run([sys.executable, "-m", "magic_model_support.scripts.prepare_qwen35_vllm_service_view",
                            "--text-model", str(model), "--multimodal-config-source", source,
                            "--output", str(target)], check=True)
            model = target
        def terminate(signum, frame):
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, terminate)
        process = subprocess.Popen(service_command(config, model, raw=args.raw), env=env)
        try:
            code = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        raise SystemExit(code)


if __name__ == "__main__":
    main()
