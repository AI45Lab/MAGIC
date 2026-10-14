"""Alternate MAGIC defender and attacker updates using modern VERL."""

import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import tempfile

import httpx

from magic_model_support.serve import runtime_environment
from magic_model_support.training.config import load_settings


def write_state(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def stop_service(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run(settings, resume=False):
    output = Path(settings["output_dir"])
    config_path, state_path = output / "config.json", output / "state.json"
    if resume:
        if json.loads(config_path.read_text()) != settings:
            raise ValueError("Resume requires the same configuration")
        state = json.loads(state_path.read_text())
    else:
        output.mkdir(parents=True, exist_ok=False)
        config_path.write_text(json.dumps(settings, indent=2) + "\n")
        state = {"completed_steps": 0, "attacker_steps": 0, "defender_steps": 0,
                 "attacker_model": settings["attacker_model"], "defender_model": settings["defender_model"]}
        write_state(state_path, state)
    family = settings["model_family"]
    env = runtime_environment(family)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, settings["active_gpus"]))
    if family == "qwen":
        env["MAGIC_QWEN35_OUTER_CONFIG_PATH"] = str(Path(settings["qwen_config_source"]) / "config.json")
        env["MAGIC_RESPONSE_ONLY_LOGITS"] = "1"
    block_size = settings["switch_steps"]
    while state["completed_steps"] < settings["total_steps"]:
        role = "defender" if (state["completed_steps"] // block_size) % 2 == 0 else "attacker"
        opponent = "attacker" if role == "defender" else "defender"
        target = state[f"{role}_steps"] + block_size
        global_target = state["completed_steps"] + block_size
        served_name = f"magic-opponent-{global_target}"
        service = {**settings["opponent_service"], "model_family": family, "model_path": state[f"{opponent}_model"],
                   "served_model_name": served_name,
                   "max_model_len": settings["max_prompt_length"] + settings["max_response_length"]}
        if family == "qwen":
            service["qwen_config_source"] = settings["qwen_config_source"]
        service_path = output / "opponent.json"
        service_path.write_text(json.dumps(service, indent=2) + "\n")
        base_url = f"http://127.0.0.1:{service['port']}"
        block_settings = {**settings, "opponent_endpoint": {"base_url": base_url + "/v1", "model": served_name}}
        block_config = output / "block.json"
        block_config.write_text(json.dumps(block_settings, indent=2) + "\n")
        role_root = output / "checkpoints" / role
        tracker = role_root / "latest_checkpointed_iteration.txt"
        saved_step = int(tracker.read_text()) if tracker.is_file() else 0
        if saved_step > target:
            raise ValueError("Checkpoint progress is ahead of the requested role block")
        if saved_step < target:
            log_path = output / f"opponent-{global_target}.log"
            with log_path.open("a") as log:
                process = subprocess.Popen([sys.executable, "-m", "magic_model_support.serve", "--config", str(service_path), "--raw"],
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    deadline = time.monotonic() + settings.get("service_start_timeout", 1800)
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError(f"Opponent service exited; see {log_path}")
                        try:
                            response = httpx.get(base_url + "/v1/models", timeout=5)
                            if response.status_code == 200 and any(item["id"] == served_name for item in response.json()["data"]):
                                break
                        except (httpx.ConnectError, httpx.TimeoutException):
                            pass
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f"Opponent startup timed out; see {log_path}")
                        time.sleep(1)
                    active = subprocess.Popen([sys.executable, "-m", "magic_model_support.training.block",
                                               "--config", str(block_config), "--role", role, "--target-step", str(target)],
                                              env=env, start_new_session=True)
                    try:
                        if active.wait() != 0:
                            raise RuntimeError(f"{role} training block failed")
                    finally:
                        stop_service(active)
                finally:
                    stop_service(process)
        checkpoint = role_root / f"global_step_{target}" / "actor"
        export = output / "models" / f"{role}-{global_target}"
        export.parent.mkdir(exist_ok=True)
        if not export.exists():
            with tempfile.TemporaryDirectory(prefix=f".{role}-", dir=export.parent) as temporary:
                subprocess.run([sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
                                "--local_dir", str(checkpoint), "--target_dir", temporary], env=env, check=True)
                for filename in ("SYSTEM_PROMPT.txt", "tekken.json"):
                    source = Path(settings[f"{role}_model"]) / filename
                    if source.is_file():
                        shutil.copyfile(source, Path(temporary) / filename)
                if family == "ministral":
                    subprocess.run([sys.executable, "-m", "magic_model_support.compat.normalize_ministral3_merged",
                                    "--model-dir", temporary], check=True)
                Path(temporary).rename(export)
        state.update({"completed_steps": global_target, f"{role}_steps": target, f"{role}_model": str(export)})
        write_state(state_path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    print(json.dumps(run(load_settings(args.config), resume=args.resume), indent=2))


if __name__ == "__main__":
    main()
