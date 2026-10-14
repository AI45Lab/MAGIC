"""Train a Qwen or Ministral SFT model with Hugging Face Trainer."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from magic_model_support.sft import tokenize_sft_record, _mistral3_checkpoint_has_separate_lm_head


class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, family, max_length):
        records = json.loads(Path(path).read_text())
        if not isinstance(records, list) or not records:
            raise ValueError("SFT input must be a nonempty JSON array")
        self.rows = []
        for i, record in enumerate(records):
            prompt, full = tokenize_sft_record(record, tokenizer, model_family=family)
            if len(full) > max_length:
                raise ValueError(f"SFT record {i} exceeds max_length={max_length}")
            if len(full) <= len(prompt):
                raise ValueError(f"SFT record {i} has no assistant tokens")
            self.rows.append({"input_ids": full, "labels": [-100] * len(prompt) + full[len(prompt):]})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class SFTCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        width = max(len(row["input_ids"]) for row in rows)
        return {
            "input_ids": torch.tensor([r["input_ids"] + [self.pad_token_id] * (width - len(r["input_ids"])) for r in rows]),
            "attention_mask": torch.tensor([[1] * len(r["input_ids"]) + [0] * (width - len(r["input_ids"])) for r in rows]),
            "labels": torch.tensor([r["labels"] + [-100] * (width - len(r["labels"])) for r in rows]),
        }


def train(config):
    family = config["model_family"]
    args = TrainingArguments(**config["training_args"])
    model_config = AutoConfig.from_pretrained(config["model_path"])
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if family == "qwen":
        if model_config.model_type != "qwen3_5_text":
            raise ValueError("Export the Qwen text checkpoint with prepare_qwen35_text_only_view before SFT")
        model = AutoModelForCausalLM.from_pretrained(config["model_path"], dtype=dtype)
    elif family == "ministral":
        from transformers import Mistral3ForConditionalGeneration
        if model_config.model_type != "mistral3":
            raise ValueError("Expected a Ministral Mistral3 checkpoint")
        if _mistral3_checkpoint_has_separate_lm_head(config["model_path"]):
            model_config.tie_word_embeddings = False
            model_config.text_config.tie_word_embeddings = False
        model = Mistral3ForConditionalGeneration.from_pretrained(config["model_path"], config=model_config, dtype=dtype)
        # Text batches never use these parameters; exclude them from optimization.
        for name, parameter in model.named_parameters():
            if "language_model" not in name and "lm_head" not in name:
                parameter.requires_grad_(False)
        model.config.text_config.use_cache = False
    else:
        raise ValueError(f"Unsupported model_family: {family}")
    model.config.use_cache = False
    # Trainer restores state_dict keys directly; save runtime names for exact reload.
    model.save_pretrained = partial(model.save_pretrained, save_original_format=False)
    training = SFTDataset(config["train_file"], tokenizer, family, config["max_length"])
    validation = (SFTDataset(config["validation_file"], tokenizer, family, config["max_length"])
                  if config.get("validation_file") else None)
    trainer = Trainer(model=model, args=args, processing_class=tokenizer,
                      train_dataset=training, eval_dataset=validation,
                      data_collator=SFTCollator(tokenizer.pad_token_id))
    trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
    # Trainer handles full FSDP state collection for save_model.
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(str(Path(args.output_dir) / "hf_model"))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(Path(args.output_dir) / "hf_model")
        for filename in ("SYSTEM_PROMPT.txt", "tekken.json"):
            source = Path(config["model_path"]) / filename
            if source.is_file():
                import shutil
                shutil.copyfile(source, Path(args.output_dir) / "hf_model" / filename)
    return trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    train(json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
