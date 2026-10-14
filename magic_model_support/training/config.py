"""Translate public MAGIC settings to the pinned VERL configuration."""

import importlib.util
import json
from pathlib import Path


DEFAULTS = {
    # Algorithm defaults from scripts/rl/separated/grpo_public.sh.
    "total_steps": 300, "switch_steps": 15, "train_batch_size": 64,
    "mini_batch_size": 32, "rollout_n": 4, "learning_rate": 1e-6,
    "max_prompt_length": 8192, "max_response_length": 6144,
    "harm_reward": 1.0, "refusal_reward": 0.5, "format_reward": 1.0,
}


def load_settings(path):
    settings = {**DEFAULTS, **json.loads(Path(path).read_text())}
    if settings["model_family"] not in ("qwen", "ministral"):
        raise ValueError("model_family must be qwen or ministral")
    if settings["switch_steps"] <= 0 or settings["total_steps"] <= 0 or settings["total_steps"] % (2 * settings["switch_steps"]):
        raise ValueError("total_steps must be a positive multiple of two role blocks")
    if settings["rollout_n"] < 2:
        raise ValueError("GRPO needs at least two samples per prompt")
    active, opponent = settings["active_gpus"], settings["opponent_service"]["cuda_visible_devices"]
    if (not active or not opponent or set(map(str, active)) & set(map(str, opponent))
            or len(set(map(str, active))) != len(active) or len(set(map(str, opponent))) != len(opponent)):
        raise ValueError("Active and opponent GPU lists must be nonempty and disjoint")
    if settings["model_family"] == "ministral" and not settings.get("system_prompt_file"):
        raise ValueError("Ministral Reasoning requires its official system_prompt_file")
    for name in ("train_file", "validation_file"):
        settings[name] = str(Path(settings[name]).resolve(strict=True))
    for name in ("attacker_model", "defender_model"):
        settings[name] = str(Path(settings[name]).resolve(strict=True))
        if not (Path(settings[name]) / "config.json").is_file():
            raise ValueError(f"{name} must contain config.json")
    settings["output_dir"] = str(Path(settings["output_dir"]).resolve())
    if settings.get("system_prompt_file"):
        settings["system_prompt_file"] = str(Path(settings["system_prompt_file"]).resolve(strict=True))
    if settings["model_family"] == "qwen":
        settings["qwen_config_source"] = str(Path(settings["qwen_config_source"]).resolve(strict=True))
    return settings


def compose_config(settings, role, target_step, config_dir=None):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    if config_dir is None:
        package = Path(next(iter(importlib.util.find_spec("verl").submodule_search_locations)))
        config_dir = package / "trainer/config"
    overrides = settings.get("verl_overrides", [])
    with initialize_config_dir(config_dir=str(Path(config_dir).resolve()), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    values = {
        "trainer.use_v1": False, "trainer.nnodes": 1,
        "trainer.n_gpus_per_node": len(settings["active_gpus"]),
        "trainer.project_name": "magic", "trainer.experiment_name": role,
        "trainer.logger": ["console"], "trainer.total_epochs": settings["total_steps"],
        "trainer.total_training_steps": target_step,
        "trainer.resume_mode": "auto", "trainer.val_before_train": False,
        "trainer.save_freq": settings["switch_steps"], "trainer.test_freq": -1,
        "trainer.default_local_dir": str(Path(settings["output_dir"]) / "checkpoints" / role),
        "data.train_files": [settings["train_file"]], "data.val_files": [settings["validation_file"]],
        "data.train_batch_size": settings["train_batch_size"], "data.return_raw_chat": True,
        "data.max_prompt_length": settings["max_prompt_length"], "data.max_response_length": settings["max_response_length"],
        "data.filter_overlong_prompts": False, "data.truncation": "error",
        "actor_rollout_ref.model.path": settings[f"{role}_model"],
        "actor_rollout_ref.model.use_remove_padding": False,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.model.override_config.use_cache": False,
        "actor_rollout_ref.actor.optim.lr": settings["learning_rate"],
        "actor_rollout_ref.actor.optim.lr_scheduler_type": "constant",
        "actor_rollout_ref.actor.ppo_mini_batch_size": settings["mini_batch_size"],
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": settings["micro_batch_size_per_gpu"],
        "actor_rollout_ref.actor.use_dynamic_bsz": False,
        "actor_rollout_ref.actor.ppo_epochs": 1, "actor_rollout_ref.actor.clip_ratio": 0.2,
        "actor_rollout_ref.actor.clip_ratio_low": 0.2, "actor_rollout_ref.actor.clip_ratio_high": 0.2,
        "actor_rollout_ref.actor.use_kl_loss": False, "actor_rollout_ref.actor.entropy_coeff": 0,
        "actor_rollout_ref.actor.entropy_from_logits_with_chunking": True,
        "actor_rollout_ref.actor.checkpoint.save_contents": ["model", "optimizer", "extra"],
        "actor_rollout_ref.actor.checkpoint.load_contents": ["model", "optimizer", "extra"],
        "actor_rollout_ref.rollout.name": "vllm", "actor_rollout_ref.rollout.mode": "async",
        "actor_rollout_ref.rollout.n": settings["rollout_n"], "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.top_p": 1.0, "actor_rollout_ref.rollout.top_k": -1,
        "actor_rollout_ref.rollout.prompt_length": settings["max_prompt_length"],
        "actor_rollout_ref.rollout.response_length": settings["max_response_length"],
        "actor_rollout_ref.rollout.max_model_len": settings["max_prompt_length"] + settings["max_response_length"],
        "actor_rollout_ref.rollout.calculate_log_probs": True,
        "actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only": True,
        "actor_rollout_ref.rollout.agent.default_agent_loop": "magic",
        "actor_rollout_ref.rollout.agent.agent_loop_config_path": str(Path(__file__).with_name("agent.yaml")),
        "actor_rollout_ref.rollout.agent.agent_loop_manager_class": "magic_model_support.training.manager.MagicAgentLoopManager",
        "algorithm.adv_estimator": "grpo", "algorithm.use_kl_in_reward": False,
        "critic.enable": False, "reward.reward_model.enable": False,
        "ray_kwargs.ray_init.address": "local",
    }
    if settings["model_family"] == "ministral":
        values.update({
            "actor_rollout_ref.actor.fsdp_config.use_orig_params": True,
            "actor_rollout_ref.model.override_config.tie_word_embeddings": False,
            "actor_rollout_ref.model.override_config.text_config.tie_word_embeddings": False,
            "actor_rollout_ref.model.override_config.text_config.architectures": ["MistralForCausalLM"],
            "actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.text_config.architectures": ["MistralForCausalLM"],
        })
    for key, value in values.items():
        OmegaConf.update(config, key, value, force_add=True)
    return config
