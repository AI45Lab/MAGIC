from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.distributed.fsdp import (
    CPUOffload,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


@dataclass(frozen=True)
class TrainConfig:
    model_path: str
    train_file: str
    validation_file: str
    output_dir: str
    cutoff_len: int = 8192
    epochs: int = 3
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1.0e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    log_steps: int = 5
    save_steps: int = 100
    eval_steps: int = 100
    seed: int = 42
    num_workers: int = 4
    cpu_offload: bool = False
    max_train_steps: int | None = None
    model_loader: str = "auto"
    chat_template_mode: str = "auto"
    chat_template_kwargs_json: str = "{}"
    max_unsharded_weight_gib: float = 100.0
    save_total_limit: int = 1


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Full-language-model SFT for MAGIC with native PyTorch FSDP.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cutoff-len", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--max-train-steps", type=int, default=None)
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
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=1,
        help="Keep only this many completed checkpoint-* directories (0 keeps all).",
    )
    return TrainConfig(**vars(parser.parse_args()))


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _is_main() -> bool:
    return _rank() == 0


def _log(message: str) -> None:
    if _is_main():
        print(message, flush=True)


def _read_records(path: str | Path) -> list[dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON array in {path}.")
    required = {"instruction", "input", "output", "system"}
    for index, record in enumerate(value):
        if not isinstance(record, dict) or not required.issubset(record):
            raise ValueError(f"Malformed SFT record {index} in {path}.")
    return value


def _messages(record: dict[str, Any]) -> list[dict[str, str]]:
    prefix: list[dict[str, str]] = []
    if str(record.get("system") or "").strip():
        prefix.append({"role": "system", "content": str(record["system"]).strip()})
    user = str(record["instruction"]).strip()
    query = str(record.get("input") or "").strip()
    if query:
        user = f"{user}\n\n{query}"
    prefix.append({"role": "user", "content": user})
    return prefix


def _qwen_chatml_literal(record: dict[str, Any], tokenizer: Any) -> tuple[list[int], list[int]]:
    """Render Qwen ChatML without its optional implicit thinking prefill."""

    prefix = "".join(
        f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        for message in _messages(record)
    )
    assistant_prefix = f"{prefix}<|im_start|>assistant\n"
    full = f"{assistant_prefix}{str(record['output']).strip()}<|im_end|>\n"
    prefix_ids = tokenizer(assistant_prefix, add_special_tokens=False)["input_ids"]
    input_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    return prefix_ids, input_ids


def _tokenizer_chat_template(
    record: dict[str, Any], tokenizer: Any, template_kwargs: dict[str, Any]
) -> tuple[list[int], list[int]]:
    """Render a model-native template and prove that the assistant mask is aligned."""

    if not hasattr(tokenizer, "apply_chat_template"):
        raise TypeError(f"Tokenizer {type(tokenizer).__name__} does not implement apply_chat_template().")
    prompt_messages = _messages(record)
    output = str(record["output"]).strip()
    full_messages = [*prompt_messages, {"role": "assistant", "content": output}]
    prefix_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
        **template_kwargs,
    )
    if output not in full_text:
        raise ValueError("The tokenizer chat template did not preserve the literal MAGIC assistant output.")
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    input_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if input_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError(
            "The generation-prompt tokenization is not a strict prefix of the completed chat. "
            "Add a model-family template adapter instead of using an incorrect assistant loss mask."
        )
    return prefix_ids, input_ids


def _has_qwen_chatml_tokens(tokenizer: Any) -> bool:
    try:
        vocab = tokenizer.get_vocab()
    except (AttributeError, TypeError):
        return False
    return "<|im_start|>" in vocab and "<|im_end|>" in vocab


def _parse_template_kwargs(value: str | dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("chat_template_kwargs_json must decode to a JSON object.")
    return parsed


def _tokenize_record(
    record: dict[str, Any],
    tokenizer: Any,
    template_mode: str = "auto",
    template_kwargs: str | dict[str, Any] | None = None,
) -> tuple[list[int], list[int]]:
    """Tokenize MAGIC while masking every token before the assistant response."""

    kwargs = _parse_template_kwargs(template_kwargs)
    if template_mode == "qwen_chatml_literal":
        return _qwen_chatml_literal(record, tokenizer)
    if template_mode not in ("auto", "tokenizer"):
        raise ValueError(f"Unknown chat template mode: {template_mode}")
    try:
        return _tokenizer_chat_template(record, tokenizer, kwargs)
    except Exception as exc:
        if template_mode == "auto" and _has_qwen_chatml_tokens(tokenizer):
            _log(
                "Tokenizer template rendering failed; using the audited Qwen ChatML literal fallback. "
                f"Original error: {type(exc).__name__}: {exc}"
            )
            return _qwen_chatml_literal(record, tokenizer)
        raise RuntimeError(
            "Model-native chat template rendering failed. The reusable image needs Jinja2>=3.1, "
            "and this model may require explicit --chat-template-kwargs-json."
        ) from exc


class MagicSFTDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        path: str | Path,
        tokenizer: Any,
        cutoff_len: int,
        template_mode: str = "auto",
        template_kwargs: str | dict[str, Any] | None = None,
    ) -> None:
        self.records = _read_records(path)
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len
        self.template_mode = template_mode
        self.template_kwargs = _parse_template_kwargs(template_kwargs)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        prefix_ids, input_ids = _tokenize_record(
            self.records[index],
            self.tokenizer,
            self.template_mode,
            self.template_kwargs,
        )
        if len(input_ids) > self.cutoff_len:
            input_ids = input_ids[: self.cutoff_len]
        if len(prefix_ids) >= len(input_ids):
            raise ValueError(
                f"Sample {index} has no supervised assistant tokens after cutoff_len={self.cutoff_len}."
            )
        labels = [-100] * min(len(prefix_ids), len(input_ids)) + input_ids[len(prefix_ids) :]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class Collator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].numel() for item in features)
        input_ids = torch.full((len(features), max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(features), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(features), max_len), dtype=torch.long)
        for row, item in enumerate(features):
            size = item["input_ids"].numel()
            input_ids[row, :size] = item["input_ids"]
            labels[row, :size] = item["labels"]
            attention_mask[row, :size] = 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def _load_qwen35_multimodal_text(model_path: str, local_rank: int) -> tuple[Any, Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    multimodal_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if getattr(multimodal_config, "model_type", None) != "qwen3_5":
        raise ValueError(f"Expected qwen3_5 checkpoint, found {getattr(multimodal_config, 'model_type', None)}")
    text_config = multimodal_config.text_config
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(text_config, trust_remote_code=True)

    index_path = Path(model_path) / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing sharded safetensors index: {index_path}")
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    wanted = {
        full_key: shard
        for full_key, shard in weight_map.items()
        if full_key.startswith("model.language_model.")
    }
    if "lm_head.weight" not in weight_map:
        raise KeyError("The multimodal checkpoint does not contain lm_head.weight.")
    wanted["lm_head.weight"] = weight_map["lm_head.weight"]
    parameter_names = set(dict(model.named_parameters()))
    buffer_names = set(dict(model.named_buffers()))
    nonpersistent_buffers: set[str] = set()
    for module_name, module in model.named_modules():
        for buffer_name in module._non_persistent_buffers_set:
            nonpersistent_buffers.add(f"{module_name}.{buffer_name}" if module_name else buffer_name)
    expected_keys = parameter_names | (buffer_names - nonpersistent_buffers)
    missing_keys = set(expected_keys)
    unexpected: list[str] = []
    tensor_by_name = dict(model.named_parameters()) | dict(model.named_buffers())
    for shard in sorted(set(wanted.values())):
        shard_state = load_file(str(Path(model_path) / shard), device="cpu")
        for full_key, filename in wanted.items():
            if filename != shard or full_key not in shard_state:
                continue
            key = full_key.replace("model.language_model.", "model.", 1)
            if full_key == "lm_head.weight":
                key = "lm_head.weight"
            if key not in tensor_by_name:
                unexpected.append(key)
                continue
            target = tensor_by_name[key]
            source = shard_state[full_key]
            if target.shape != source.shape:
                raise RuntimeError(f"Shape mismatch for {key}: expected={target.shape}, found={source.shape}")
            module_path, _, tensor_name = key.rpartition(".")
            module = model.get_submodule(module_path) if module_path else model
            value = source.to(device=torch.device("cuda", local_rank), dtype=source.dtype)
            if key in parameter_names:
                setattr(module, tensor_name, torch.nn.Parameter(value, requires_grad=True))
            else:
                setattr(module, tensor_name, value)
            tensor_by_name[key] = getattr(module, tensor_name)
            missing_keys.discard(key)
        del shard_state
    if missing_keys or unexpected:
        raise RuntimeError(
            f"Text-only checkpoint conversion mismatch: missing={sorted(missing_keys)[:8]}, "
            f"unexpected={unexpected[:8]}"
        )
    # RoPE frequencies are intentionally non-persistent and absent from safetensors.
    # Recompute them from the text config on each rank after loading trainable weights.
    for module in model.modules():
        if module.__class__.__name__.endswith("RotaryEmbedding"):
            inv_freq, attention_scaling = module.compute_default_rope_parameters(
                module.config, torch.device("cuda", local_rank)
            )
            module.inv_freq = inv_freq
            module.original_inv_freq = inv_freq.clone()
            module.attention_scaling = attention_scaling
    model.config.use_cache = False
    if not model.config.architectures:
        model.config.architectures = [model.__class__.__name__]
    return model, tokenizer, text_config


def _is_expert_model_config(config: Any) -> bool:
    for name in ("n_routed_experts", "num_local_experts", "num_experts"):
        value = getattr(config, name, 0)
        if isinstance(value, int) and value > 1:
            return True
    return False


def _is_qwen35_multimodal_config(config: Any) -> bool:
    return (
        getattr(config, "model_type", None) == "qwen3_5"
        and hasattr(config, "text_config")
        and not getattr(config, "language_model_only", False)
    )


def _is_mistral3_multimodal_config(config: Any) -> bool:
    """Identify the Mistral3 wrapper used by Ministral-3 multimodal checkpoints."""

    return getattr(config, "model_type", None) == "mistral3" and hasattr(config, "text_config")


def _resolve_model_loader(config: Any, loader_mode: str) -> str:
    if loader_mode != "auto":
        return loader_mode
    if _is_qwen35_multimodal_config(config):
        return "qwen3_5_multimodal_text"
    if _is_mistral3_multimodal_config(config):
        return "multimodal_causal_lm"
    return "generic_causal_lm"


def _checkpoint_weight_bytes(model_path: str | Path) -> int | None:
    root = Path(model_path)
    index = root / "model.safetensors.index.json"
    if index.is_file():
        payload = json.loads(index.read_text(encoding="utf-8"))
        total_size = payload.get("metadata", {}).get("total_size")
        if isinstance(total_size, int) and total_size > 0:
            return total_size
        shards = {root / value for value in payload.get("weight_map", {}).values()}
        if shards and all(path.is_file() for path in shards):
            return sum(path.stat().st_size for path in shards)
    weights = list(root.glob("*.safetensors"))
    if weights:
        return sum(path.stat().st_size for path in weights)
    return None


def _mistral3_checkpoint_has_separate_lm_head(model_path: str | Path) -> bool:
    """Return whether the native or converted index stores both embeddings and LM head."""

    index = Path(model_path) / "model.safetensors.index.json"
    if not index.is_file():
        return False
    weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
    native_pair = {
        "language_model.lm_head.weight",
        "language_model.model.embed_tokens.weight",
    }
    hf_pair = {
        "lm_head.weight",
        "model.language_model.embed_tokens.weight",
    }
    keys = set(weight_map)
    return native_pair.issubset(keys) or hf_pair.issubset(keys)


def _load_generic_causal_model(
    model_path: str, local_rank: int, max_unsharded_weight_gib: float
) -> tuple[Any, Any, Any]:
    """Load a text-only HF causal LM directly on each rank before FSDP shards it."""

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer defines neither pad_token_id nor eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(config, "text_config") and not getattr(config, "language_model_only", False):
        raise ValueError(
            "generic_causal_lm cannot silently discard a multimodal tower; add an audited family loader."
        )
    if _is_expert_model_config(config):
        raise ValueError(
            "generic_causal_lm intentionally rejects MoE checkpoints. Expert models "
            "need an audited tensor/expert-parallel training design."
        )
    weight_bytes = _checkpoint_weight_bytes(model_path)
    weight_limit = int(max_unsharded_weight_gib * 1024**3)
    if weight_bytes is not None and weight_bytes > weight_limit:
        raise ValueError(
            f"Checkpoint weights are {weight_bytes / 1024**3:.1f} GiB, above the "
            f"{max_unsharded_weight_gib:.1f} GiB per-rank pre-FSDP loading limit. "
            "Use a genuinely sharded initialization path instead of exceeding device memory."
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": local_rank},
    )
    model.config.use_cache = False
    if not model.config.architectures:
        model.config.architectures = [model.__class__.__name__]
    return model, tokenizer, config


def _load_multimodal_causal_model(
    model_path: str, local_rank: int, max_unsharded_weight_gib: float
) -> tuple[Any, Any, Any]:
    """Load an audited HF conditional-generation wrapper for text-only batches.

    This keeps the full wrapper/checkpoint so model-native text forwarding remains intact.
    The unused vision tower receives no gradients because MAGIC examples contain no images.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer defines neither pad_token_id nor eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if not hasattr(config, "text_config"):
        raise ValueError("multimodal_causal_lm requires a checkpoint with text_config.")
    if _is_expert_model_config(config) or _is_expert_model_config(config.text_config):
        raise ValueError("multimodal_causal_lm does not support expert-parallel checkpoints.")
    weight_bytes = _checkpoint_weight_bytes(model_path)
    if weight_bytes is not None and weight_bytes > max_unsharded_weight_gib * 1024**3:
        raise ValueError(
            f"Checkpoint weights are {weight_bytes / 1024**3:.1f} GiB, above the "
            f"{max_unsharded_weight_gib:.1f} GiB per-rank loading limit."
        )
    if not _is_mistral3_multimodal_config(config):
        raise ValueError(
            "multimodal_causal_lm is currently audited only for Mistral3ForConditionalGeneration."
        )
    # Mistral3Config is intentionally absent from AutoModelForCausalLM: its native
    # wrapper accepts both text and image inputs and is registered as conditional
    # generation. Loading the native class preserves checkpoint key names and lets
    # text-only MAGIC batches omit pixel_values safely.
    from transformers import Mistral3ForConditionalGeneration

    # This Ministral release stores independent embedding and lm_head tensors even
    # though the composite config inherits tie_word_embeddings=True. Preserve the
    # checkpoint's actual parameterization and avoid a misleading tied-weight load.
    if _mistral3_checkpoint_has_separate_lm_head(model_path):
        config.tie_word_embeddings = False
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": local_rank},
    )
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    return model, tokenizer, config


def _load_model(
    model_path: str,
    local_rank: int,
    loader_mode: str,
    max_unsharded_weight_gib: float = 100.0,
) -> tuple[Any, Any, Any, str]:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    resolved = _resolve_model_loader(config, loader_mode)
    if resolved == "qwen3_5_multimodal_text":
        model, tokenizer, text_config = _load_qwen35_multimodal_text(model_path, local_rank)
    elif resolved == "generic_causal_lm":
        model, tokenizer, text_config = _load_generic_causal_model(
            model_path, local_rank, max_unsharded_weight_gib
        )
    elif resolved == "multimodal_causal_lm":
        model, tokenizer, text_config = _load_multimodal_causal_model(
            model_path, local_rank, max_unsharded_weight_gib
        )
    else:
        raise ValueError(f"Unknown model loader mode: {loader_mode}")
    return model, tokenizer, text_config, resolved


def _decoder_layer_classes(model: torch.nn.Module) -> set[type[torch.nn.Module]]:
    classes = {
        type(module)
        for module in model.modules()
        if module.__class__.__name__.endswith("DecoderLayer")
    }
    if not classes:
        raise RuntimeError("Could not locate a decoder-layer class for FSDP auto wrapping.")
    return classes


def _optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if name.endswith("bias") or "norm" in name.lower() else decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        fused=True,
    )


def _loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1, :].contiguous().float().view(-1, logits.size(-1)),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def evaluate(model: FSDP, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = torch.zeros(2, dtype=torch.float64, device=device)
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        valid = (batch["labels"][:, 1:] != -100).sum()
        total[0] += _loss(output.logits, batch["labels"]).double() * valid
        total[1] += valid
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    model.train()
    return (total[0] / total[1].clamp_min(1)).item()


def _save_checkpoint(
    model: FSDP,
    scheduler: Any,
    tokenizer: Any,
    text_config: Any,
    cfg: TrainConfig,
    global_step: int,
) -> None:
    checkpoint = Path(cfg.output_dir) / f"checkpoint-{global_step}"
    hf_dir = checkpoint / "hf_model"
    if _is_main():
        checkpoint.mkdir(parents=True, exist_ok=True)
        hf_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, state_cfg):
        model_state = model.state_dict()
    if _is_main():
        text_config.save_pretrained(hf_dir)
        tokenizer.save_pretrained(hf_dir)
        save_file(model_state, str(hf_dir / "model.safetensors"), metadata={"format": "pt"})
        torch.save(
            {
                "scheduler": scheduler.state_dict(),
                "global_step": global_step,
                "config": asdict(cfg),
                "resume_supported": False,
            },
            checkpoint / "trainer_state.pt",
        )
        (Path(cfg.output_dir) / "latest_checkpoint.txt").write_text(str(checkpoint), encoding="utf-8")
        if cfg.save_total_limit > 0:
            candidates: list[tuple[int, Path]] = []
            for path in Path(cfg.output_dir).glob("checkpoint-*"):
                match = re.fullmatch(r"checkpoint-(\d+)", path.name)
                if match and path.is_dir():
                    candidates.append((int(match.group(1)), path))
            candidates.sort(reverse=True)
            keep = {checkpoint}
            for _, path in candidates:
                if path != checkpoint and len(keep) < cfg.save_total_limit:
                    keep.add(path)
            for _, stale in candidates:
                if stale not in keep:
                    shutil.rmtree(stale)
    del model_state
    dist.barrier()


def main() -> None:
    cfg = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the MAGIC FSDP SFT job.")
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    random.seed(cfg.seed + _rank())
    torch.manual_seed(cfg.seed + _rank())
    torch.cuda.manual_seed_all(cfg.seed + _rank())

    model, tokenizer, text_config, resolved_loader = _load_model(
        cfg.model_path, local_rank, cfg.model_loader, cfg.max_unsharded_weight_gib
    )
    decoder_classes = _decoder_layer_classes(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = FSDP(
        model,
        auto_wrap_policy=lambda module, recurse, nonwrapped_numel: transformer_auto_wrap_policy(
            module,
            recurse,
            nonwrapped_numel,
            transformer_layer_cls=decoder_classes,
        ),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        ),
        cpu_offload=CPUOffload(offload_params=cfg.cpu_offload),
        device_id=device,
        limit_all_gathers=True,
        use_orig_params=True,
        sync_module_states=False,
    )

    collator = Collator(tokenizer.pad_token_id)
    train_dataset = MagicSFTDataset(
        cfg.train_file,
        tokenizer,
        cfg.cutoff_len,
        cfg.chat_template_mode,
        cfg.chat_template_kwargs_json,
    )
    validation_dataset = MagicSFTDataset(
        cfg.validation_file,
        tokenizer,
        cfg.cutoff_len,
        cfg.chat_template_mode,
        cfg.chat_template_kwargs_json,
    )
    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=cfg.seed, drop_last=False)
    # FSDP performs collectives during every forward. All ranks must therefore
    # execute the same number of validation batches. DistributedSampler pads at
    # most world_size - 1 examples when the validation size is not divisible by
    # world_size, keeping the number of collective calls aligned across ranks.
    validation_sampler = DistributedSampler(
        validation_dataset,
        shuffle=False,
        drop_last=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.micro_batch_size,
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.micro_batch_size,
        sampler=validation_sampler,
        collate_fn=collator,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    optimizer = _optimizer(model, cfg)
    updates_per_epoch = math.ceil(len(train_loader) / cfg.gradient_accumulation_steps)
    total_steps = cfg.epochs * updates_per_epoch
    if cfg.max_train_steps is not None:
        total_steps = min(total_steps, cfg.max_train_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(total_steps * cfg.warmup_ratio),
        num_training_steps=total_steps,
    )
    if _is_main():
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(cfg.output_dir) / "resolved_config.json").write_text(
            json.dumps(asdict(cfg), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    _log(
        f"train={len(train_dataset)} validation={len(validation_dataset)} world_size={_world_size()} "
        f"validation_padding={len(validation_sampler) * _world_size() - len(validation_dataset)} "
        f"global_batch={cfg.micro_batch_size * _world_size() * cfg.gradient_accumulation_steps} "
        f"total_steps={total_steps} model_loader={resolved_loader} "
        f"chat_template_mode={cfg.chat_template_mode} "
        f"decoder_classes={[x.__name__ for x in decoder_classes]}"
    )

    model.train()
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(cfg.epochs):
        train_sampler.set_epoch(epoch)
        for micro_step, batch in enumerate(train_loader):
            last_micro_step = micro_step + 1 == len(train_loader)
            should_update = (micro_step + 1) % cfg.gradient_accumulation_steps == 0 or last_micro_step
            sync_context = nullcontext() if should_update else model.no_sync()
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with sync_context:
                output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                divisor = cfg.gradient_accumulation_steps
                if last_micro_step and (micro_step + 1) % cfg.gradient_accumulation_steps:
                    divisor = (micro_step % cfg.gradient_accumulation_steps) + 1
                loss = _loss(output.logits, batch["labels"]) / divisor
                loss.backward()
            if not should_update:
                continue
            grad_norm = model.clip_grad_norm_(cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step % cfg.log_steps == 0:
                reduced = loss.detach() * divisor
                dist.all_reduce(reduced, op=dist.ReduceOp.AVG)
                _log(
                    f"step={global_step}/{total_steps} epoch={epoch + 1} "
                    f"loss={reduced.item():.6f} grad_norm={float(grad_norm):.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.3e}"
                )
            if cfg.eval_steps > 0 and global_step % cfg.eval_steps == 0:
                _log(f"step={global_step} validation_loss={evaluate(model, validation_loader, device):.6f}")
            if cfg.save_steps > 0 and global_step % cfg.save_steps == 0:
                _save_checkpoint(model, scheduler, tokenizer, text_config, cfg, global_step)
            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break
    if cfg.save_steps <= 0 or global_step % cfg.save_steps != 0:
        _save_checkpoint(model, scheduler, tokenizer, text_config, cfg, global_step)
    _log(f"Training complete at step={global_step}.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
