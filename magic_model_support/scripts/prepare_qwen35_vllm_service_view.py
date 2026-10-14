#!/usr/bin/env python3
"""Create a zero-copy Qwen3.5 vLLM service view around text-only weights."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _link(source: Path, target: Path) -> None:
    os.symlink(source.resolve(), target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument("--multimodal-config-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    text_model = args.text_model.resolve()
    config_source = args.multimodal_config_source.resolve()
    output = args.output

    text_config = _load_json(text_model / "config.json")
    if text_config.get("model_type") != "qwen3_5_text":
        raise ValueError(
            f"expected qwen3_5_text weights, found {text_config.get('model_type')!r}"
        )
    text_config["architectures"] = ["Qwen3_5ForCausalLM"]

    outer_config = _load_json(config_source / "config.json")
    if outer_config.get("model_type") != "qwen3_5":
        raise ValueError(
            f"expected qwen3_5 outer config, found {outer_config.get('model_type')!r}"
        )
    # Keep the outer Qwen3_5Config type required by vLLM's OpenAI renderer,
    # but select the text-only implementation whose parameter names match the
    # SFT checkpoint (model.* rather than language_model.model.*).
    outer_config["architectures"] = ["Qwen3_5ForCausalLM"]
    outer_config["text_config"] = text_config

    single_weights = text_model / "model.safetensors"
    index_path = text_model / "model.safetensors.index.json"
    if single_weights.is_file():
        weight_files = [single_weights]
        tensor_count = None
        total_size = single_weights.stat().st_size
    elif index_path.is_file():
        index = _load_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid weight_map in {index_path}")
        weight_files = [text_model / name for name in sorted(set(weight_map.values()))]
        tensor_count = len(weight_map)
        total_size = index.get("metadata", {}).get("total_size")
    else:
        raise FileNotFoundError(f"no safetensors checkpoint found in {text_model}")
    for path in weight_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    output.mkdir(parents=True, exist_ok=False)
    for path in weight_files:
        _link(path, output / path.name)
    if index_path.is_file():
        _link(index_path, output / index_path.name)

    metadata_names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
        "generation_config.json",
        "preprocessor_config.json",
        "added_tokens.json",
        "merges.txt",
        "vocab.json",
    )
    for name in metadata_names:
        source = text_model / name
        if source.is_file():
            _link(source, output / name)

    (output / "config.json").write_text(
        json.dumps(outer_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "text_model": str(text_model),
        "multimodal_config_source": str(config_source),
        "output": str(output.resolve()),
        "outer_architecture": outer_config["architectures"][0],
        "outer_model_type": outer_config["model_type"],
        "text_architecture": text_config["architectures"][0],
        "text_model_type": text_config["model_type"],
        "tensor_count": tensor_count,
        "total_size": total_size,
        "weight_files": len(weight_files),
    }
    (output / "service_view_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
