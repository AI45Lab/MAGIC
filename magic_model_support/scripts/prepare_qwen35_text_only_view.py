#!/usr/bin/env python3
"""Export the language-model tensors of a Qwen multimodal checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_5":
        raise ValueError(f"expected qwen3_5 source, found {config.get('model_type')!r}")
    text_config = dict(config["text_config"])
    text_config.update(architectures=["Qwen3_5ForCausalLM"], model_type="qwen3_5_text")
    source_index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))
    selected = {}
    for original, shard in source_index["weight_map"].items():
        if original.startswith("model.language_model."):
            name = "model." + original.removeprefix("model.language_model.")
        elif original == "lm_head.weight":
            name = original
        else:
            continue
        if name in selected:
            raise ValueError(f"checkpoint key collision: {name}")
        selected[name] = (original, shard)
    if "model.embed_tokens.weight" not in selected or "lm_head.weight" not in selected:
        raise ValueError("source must contain text embeddings and an LM head")

    output.mkdir(parents=True, exist_ok=False)
    total_size = 0
    weight_map = {}
    for shard in sorted({shard for _, shard in selected.values()}):
        with safe_open(source / shard, framework="pt", device="cpu") as handle:
            tensors = {
                name: handle.get_tensor(original)
                for name, (original, filename) in selected.items() if filename == shard
            }
        # Rewrite tensor names inside each shard as well as in the index.
        save_file(tensors, output / shard, metadata={"format": "pt"})
        total_size += sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
        weight_map.update({name: shard for name in tensors})
        del tensors
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "special_tokens_map.json", "generation_config.json"):
        if (source / name).is_file():
            shutil.copyfile(source / name, output / name)
    (output / "config.json").write_text(json.dumps(text_config, indent=2) + "\n")
    (output / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": total_size}, "weight_map": weight_map,
    }, indent=2) + "\n")
    print(f"Exported {len(weight_map)} text tensors to {output}")


if __name__ == "__main__":
    main()
