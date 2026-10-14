#!/usr/bin/env python3
"""Normalize VERL-merged Ministral-3 weights for the vLLM Mistral3 loader."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


OLD_PREFIX = "model.language_model.model."
NEW_PREFIX = "model.language_model."
EMBED_KEY = "model.language_model.embed_tokens.weight"
TEXT_EMBED_KEY = "model.embed_tokens.weight"
LM_HEAD_KEY = "lm_head.weight"
LM_HEAD_ALIASES = {
    "language_model.lm_head.weight",
    "model.language_model.lm_head.weight",
}


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def normalize(model_dir: Path) -> dict[str, object]:
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text())
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        locations = json.loads(index_path.read_text())["weight_map"]
    else:
        with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as handle:
            locations = {name: "model.safetensors" for name in handle.keys()}
    selected = {}
    renamed = 0
    for name, shard in locations.items():
        mapped = LM_HEAD_KEY if name in LM_HEAD_ALIASES else (NEW_PREFIX + name[len(OLD_PREFIX):] if name.startswith(OLD_PREFIX) else name)
        if mapped in selected:
            other_name, other_shard = selected[mapped]
            with safe_open(model_dir / shard, framework="pt", device="cpu") as left, safe_open(model_dir / other_shard, framework="pt", device="cpu") as right:
                if not left.get_tensor(name).equal(right.get_tensor(other_name)):
                    raise RuntimeError(f"conflicting weights map to {mapped}")
        else:
            selected[mapped] = (name, shard)
        renamed += int(mapped != name)
    if not ({EMBED_KEY, TEXT_EMBED_KEY} & selected.keys()):
        raise KeyError("missing text embedding weight")
    if LM_HEAD_KEY not in selected:
        raise RuntimeError("missing independent LM head; embeddings cannot replace the output head")
    config["tie_word_embeddings"] = False
    if "text_config" in config:
        config["text_config"]["tie_word_embeddings"] = False
    if renamed:
        total_size, weight_map = 0, {}
        with tempfile.TemporaryDirectory(prefix=".normalize-", dir=model_dir) as temporary:
            for shard in sorted({shard for _, shard in selected.values()}):
                with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
                    weights = {mapped: handle.get_tensor(name) for mapped, (name, filename) in selected.items() if filename == shard}
                save_file(weights, Path(temporary) / shard, metadata={"format": "pt"})
                total_size += sum(t.numel() * t.element_size() for t in weights.values())
                weight_map.update({name: shard for name in weights})
                del weights
            for shard in set(weight_map.values()):
                os.replace(Path(temporary) / shard, model_dir / shard)
            if index_path.is_file():
                atomic_json(index_path, {"metadata": {"total_size": total_size}, "weight_map": weight_map})
    atomic_json(config_path, config)
    result = {"model_dir": str(model_dir), "renamed_weights": renamed, "added_lm_head": False,
              "weight_count": len(selected), "status": "normalized" if renamed else "already_normalized"}
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    args = parser.parse_args()
    normalize(args.model_dir)


if __name__ == "__main__":
    main()
