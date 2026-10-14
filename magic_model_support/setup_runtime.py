"""Apply version-specific compatibility patches to a dedicated runtime."""

import argparse
import importlib.metadata
import importlib.util
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-family", choices=["qwen", "ministral"], required=True)
    args = parser.parse_args()
    if importlib.metadata.version("vllm") != "0.18.0" or importlib.metadata.version("verl") != "0.10.0.dev0":
        raise RuntimeError("Use the documented vLLM 0.18.0 / VERL 0.10.0.dev0 runtime")
    verl = Path(next(iter(importlib.util.find_spec("verl").submodule_search_locations)))
    vllm = Path(next(iter(importlib.util.find_spec("vllm").submodule_search_locations)))
    patch_dir = Path(__file__).parent / "compat/patches"
    patches = [(verl / "workers/engine/fsdp/transformer_impl.py", "verl010_no_rmpad_logprob.patch")]
    if args.model_family == "qwen":
        patches += [(verl / "workers/engine/fsdp/transformer_impl.py", "verl010_qwen35_response_logits.patch"),
                    (verl / "workers/rollout/vllm_rollout/vllm_async_server.py", "verl010_qwen35_text_renderer.patch")]
    else:
        patches += [(vllm / "model_executor/models/mistral.py", "vllm018_mistral_llama4_rope_parameters.patch")]
    for target, filename in patches:
        content = (patch_dir / filename).read_bytes()
        check = subprocess.run(["patch", "--dry-run", "--forward", "--batch", str(target)], input=content, capture_output=True)
        if check.returncode == 0:
            subprocess.run(["patch", "--forward", "--batch", str(target)], input=content, check=True)
        else:
            subprocess.run(["patch", "--dry-run", "--reverse", "--batch", str(target)], input=content, check=True)
    if args.model_family == "ministral":
        subprocess.run([sys.executable, "-m", "magic_model_support.compat.patch_vllm018_yarn_mscale",
                        "--target", str(vllm / "model_executor/layers/rotary_embedding/__init__.py")], check=True)


if __name__ == "__main__":
    main()
