from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

from magic_model_support.fsdp_sft import (
    _checkpoint_weight_bytes,
    _is_expert_model_config,
    _is_mistral3_multimodal_config,
    _is_qwen35_multimodal_config,
    _read_records,
    _resolve_model_loader,
    _tokenize_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate prepared MAGIC SFT data before model loading.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--cutoff-len", type=int, default=8192)
    parser.add_argument(
        "--model-loader",
        choices=("auto", "qwen3_5_multimodal_text", "generic_causal_lm", "multimodal_causal_lm"),
        default="auto",
    )
    parser.add_argument(
        "--chat-template-mode",
        choices=("auto", "tokenizer", "qwen_chatml_literal"),
        default="auto",
    )
    parser.add_argument("--chat-template-kwargs-json", default="{}")
    parser.add_argument("--max-unsharded-weight-gib", type=float, default=100.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    resolved_loader = _resolve_model_loader(config, args.model_loader)
    if resolved_loader == "qwen3_5_multimodal_text" and not _is_qwen35_multimodal_config(config):
        raise ValueError("qwen3_5_multimodal_text requires a multimodal qwen3_5 checkpoint.")
    if resolved_loader == "generic_causal_lm" and hasattr(config, "text_config") and not getattr(
        config, "language_model_only", False
    ):
        raise ValueError("generic_causal_lm cannot silently discard a multimodal tower.")
    if resolved_loader == "multimodal_causal_lm" and not hasattr(config, "text_config"):
        raise ValueError("multimodal_causal_lm requires a checkpoint with text_config.")
    if resolved_loader == "multimodal_causal_lm" and not _is_mistral3_multimodal_config(config):
        raise ValueError(
            "multimodal_causal_lm is currently audited only for Mistral3ForConditionalGeneration."
        )
    if resolved_loader == "multimodal_causal_lm" and (
        _is_expert_model_config(config) or _is_expert_model_config(config.text_config)
    ):
        raise ValueError("multimodal_causal_lm does not support expert-parallel checkpoints.")
    if resolved_loader == "generic_causal_lm" and _is_expert_model_config(config):
        raise ValueError("generic_causal_lm does not support MoE checkpoints; use an expert-parallel design.")
    weight_bytes = _checkpoint_weight_bytes(args.model_path)
    if (
        resolved_loader in ("generic_causal_lm", "multimodal_causal_lm")
        and weight_bytes is not None
        and weight_bytes > args.max_unsharded_weight_gib * 1024**3
    ):
        raise ValueError(
            f"Checkpoint weights are {weight_bytes / 1024**3:.1f} GiB, above the "
            f"{args.max_unsharded_weight_gib:.1f} GiB generic loader limit."
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    counts: Counter[str] = Counter()
    lengths: list[int] = []
    for split, path in (("train", args.train_file), ("validation", args.validation_file)):
        records = _read_records(path)
        counts[split] = len(records)
        for index, record in enumerate(records):
            prefix_ids, ids = _tokenize_record(
                record,
                tokenizer,
                args.chat_template_mode,
                args.chat_template_kwargs_json,
            )
            lengths.append(len(ids))
            if len(prefix_ids) >= min(len(ids), args.cutoff_len):
                raise ValueError(f"No supervised tokens after cutoff at {path}:{index}.")
            if len(ids) > args.cutoff_len:
                counts[f"{split}:truncated"] += 1
    ordered = sorted(lengths)
    summary = {
        "counts": dict(sorted(counts.items())),
        "model_type": getattr(config, "model_type", None),
        "resolved_model_loader": resolved_loader,
        "chat_template_mode": args.chat_template_mode,
        "checkpoint_weight_bytes": weight_bytes,
        "cutoff_len": args.cutoff_len,
        "token_lengths": {
            "min": ordered[0],
            "median": ordered[len(ordered) // 2],
            "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
            "max": ordered[-1],
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
