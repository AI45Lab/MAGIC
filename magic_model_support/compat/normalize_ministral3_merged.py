#!/usr/bin/env python3
"""Normalize VERL-merged Ministral-3 weights for the vLLM Mistral3 loader."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file


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
    weights_path = model_dir / "model.safetensors"
    config_path = model_dir / "config.json"
    if not weights_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"incomplete merged model: {model_dir}")
    if (model_dir / "model.safetensors.index.json").exists():
        raise RuntimeError("sharded merged safetensors are not supported by this normalizer")

    temporary_weights = weights_path.with_name(".model.safetensors.normalizing.tmp")
    temporary_weights.unlink(missing_ok=True)

    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        input_keys = set(handle.keys())

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["tie_word_embeddings"] = False
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        text_config["tie_word_embeddings"] = False

    if (
        (EMBED_KEY in input_keys or TEXT_EMBED_KEY in input_keys)
        and LM_HEAD_KEY in input_keys
        and not any(name.startswith(OLD_PREFIX) for name in input_keys)
        and not (input_keys & LM_HEAD_ALIASES)
    ):
        atomic_json(config_path, config)
        result = {
            "model_dir": str(model_dir),
            "renamed_weights": 0,
            "added_lm_head": False,
            "weight_count": len(input_keys),
            "status": "already_normalized",
        }
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result

    weights = load_file(weights_path, device="cpu")
    normalized = {}
    renamed = 0
    for name, tensor in weights.items():
        if name in LM_HEAD_ALIASES:
            new_name = LM_HEAD_KEY
        else:
            new_name = NEW_PREFIX + name[len(OLD_PREFIX) :] if name.startswith(OLD_PREFIX) else name
        if new_name in normalized:
            if not normalized[new_name].equal(tensor):
                raise RuntimeError(
                    f"conflicting weights map to {new_name}; refusing an ambiguous LM head"
                )
            renamed += int(new_name != name)
            continue
        normalized[new_name] = tensor
        renamed += int(new_name != name)

    embed_key = TEXT_EMBED_KEY if TEXT_EMBED_KEY in normalized else EMBED_KEY
    if embed_key not in normalized:
        raise KeyError(
            f"missing required embedding weight; expected {TEXT_EMBED_KEY} or {EMBED_KEY}"
        )
    added_lm_head = LM_HEAD_KEY not in normalized
    if added_lm_head:
        raise RuntimeError(
            "merged Ministral checkpoint has no independent LM head; refusing to clone "
            "embeddings because this pipeline trains an untied output head"
        )

    save_file(normalized, temporary_weights, metadata={"format": "pt"})
    with safe_open(temporary_weights, framework="pt", device="cpu") as handle:
        output_keys = set(handle.keys())
    if embed_key not in output_keys or LM_HEAD_KEY not in output_keys:
        raise RuntimeError("normalized safetensors failed required-key validation")
    if any(name.startswith(OLD_PREFIX) for name in output_keys):
        raise RuntimeError("normalized safetensors still contain the legacy nested prefix")
    if output_keys & LM_HEAD_ALIASES:
        raise RuntimeError("normalized safetensors still contain an LM-head alias")
    os.replace(temporary_weights, weights_path)

    atomic_json(config_path, config)

    result = {
        "model_dir": str(model_dir),
        "renamed_weights": renamed,
        "added_lm_head": False,
        "weight_count": len(output_keys),
        "status": "normalized",
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    args = parser.parse_args()
    normalize(args.model_dir)


if __name__ == "__main__":
    main()
