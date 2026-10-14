#!/usr/bin/env python3
"""Create a zero-copy HF text-model view of a Qwen3.5 multimodal checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path
from typing import Any


def _header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"truncated safetensors prefix: {path}")
        size = struct.unpack("<Q", raw)[0]
        payload = handle.read(size)
    if len(payload) != size:
        raise ValueError(f"truncated safetensors header: {path}")
    value = json.loads(payload.decode("utf-8"))
    return {key: item for key, item in value.items() if key != "__metadata__"}


def _tensor_bytes(metadata: dict[str, Any]) -> int:
    start, end = metadata["data_offsets"]
    return int(end) - int(start)


def _source_tensors(source: Path, weight_map: dict[str, str]) -> dict[str, dict[str, Any]]:
    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(name)
    tensors: dict[str, dict[str, Any]] = {}
    for shard, names in sorted(by_shard.items()):
        shard_header = _header(source / shard)
        for name in names:
            if name not in shard_header:
                raise KeyError(f"{name} is missing from {source / shard}")
            tensors[name] = shard_header[name]
    return tensors


def _reference_tensors(reference: Path) -> dict[str, dict[str, Any]]:
    single = reference / "model.safetensors"
    if single.exists():
        return _header(single)
    index_path = reference / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    return _source_tensors(reference, index["weight_map"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-text-model", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output
    reference = args.reference_text_model.resolve()
    source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if source_config.get("model_type") != "qwen3_5":
        raise ValueError(f"expected qwen3_5 source, found {source_config.get('model_type')!r}")
    text_config = dict(source_config["text_config"])
    text_config["architectures"] = ["Qwen3_5ForCausalLM"]
    text_config["model_type"] = "qwen3_5_text"

    source_index = json.loads(
        (source / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    original_map: dict[str, str] = source_index["weight_map"]
    renamed_map: dict[str, str] = {}
    original_for_renamed: dict[str, str] = {}
    prefix = "model.language_model."
    for original, shard in original_map.items():
        if original.startswith(prefix):
            renamed = "model." + original[len(prefix) :]
        elif original == "lm_head.weight":
            renamed = original
        else:
            continue
        if renamed in renamed_map:
            raise ValueError(f"renamed checkpoint key collision: {renamed}")
        renamed_map[renamed] = shard
        original_for_renamed[renamed] = original
    if not renamed_map:
        raise ValueError("no text-model tensors were selected")

    selected_original = {original: original_map[original] for original in original_for_renamed.values()}
    source_meta = _source_tensors(source, selected_original)
    reference_meta = _reference_tensors(reference)
    if set(renamed_map) != set(reference_meta):
        missing = sorted(set(reference_meta) - set(renamed_map))[:8]
        extra = sorted(set(renamed_map) - set(reference_meta))[:8]
        raise ValueError(f"text key mismatch against SFT reference: missing={missing}, extra={extra}")
    for renamed, original in original_for_renamed.items():
        left = source_meta[original]
        right = reference_meta[renamed]
        if left.get("shape") != right.get("shape") or left.get("dtype") != right.get("dtype"):
            raise ValueError(
                f"tensor metadata mismatch for {renamed}: source={left}, reference={right}"
            )

    output.mkdir(parents=True, exist_ok=False)
    for shard in sorted(set(renamed_map.values())):
        os.symlink(source / shard, output / shard)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
        "generation_config.json",
    ):
        source_file = source / name
        if source_file.exists():
            os.symlink(source_file, output / name)
    (output / "config.json").write_text(
        json.dumps(text_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    total_size = sum(_tensor_bytes(source_meta[name]) for name in selected_original)
    index = {"metadata": {"total_size": total_size}, "weight_map": renamed_map}
    (output / "model.safetensors.index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "source": str(source),
        "reference_text_model": str(reference),
        "output": str(output.resolve()),
        "tensor_count": len(renamed_map),
        "total_size": total_size,
        "architecture": text_config["architectures"][0],
        "model_type": text_config["model_type"],
    }
    (output / "text_view_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
