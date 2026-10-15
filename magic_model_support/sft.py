"""Model loading and assistant-token alignment for Qwen and Ministral."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


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
    if input_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("Qwen ChatML prompt tokens are not a prefix of the completed chat.")
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


def load_qwen_text_model(model_path: str, local_rank: int) -> tuple[Any, Any, Any]:
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


def tokenize_sft_record(record: dict[str, Any], tokenizer: Any, *, model_family: str, template_kwargs: dict[str, Any] | None = None) -> tuple[list[int], list[int]]:
    """Return prompt and full token IDs; mask the prompt prefix with -100 for SFT."""
    if model_family == "qwen":
        return _qwen_chatml_literal(record, tokenizer)
    if model_family == "ministral":
        return _tokenizer_chat_template(record, tokenizer, template_kwargs or {})
    raise ValueError(f"Unsupported model family: {model_family}")


def load_ministral_model(
    model_path: str, local_rank: int
) -> tuple[Any, Any, Any]:
    """Load the native HF conditional-generation wrapper for text-only batches.

    This keeps the full wrapper/checkpoint so model-native text forwarding remains intact.
    The unused vision tower receives no gradients because MAGIC examples contain no images.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer defines neither pad_token_id nor eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if config.model_type != "mistral3":
        raise ValueError(f"Expected mistral3 checkpoint, found {config.model_type}")
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
        config.text_config.tie_word_embeddings = False
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
