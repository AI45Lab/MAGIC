import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from magic_model_support.sft import (
    _mistral3_checkpoint_has_separate_lm_head,
    load_ministral_model,
    tokenize_sft_record,
)


class TinyTokenizer:
    def __call__(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": list(text.encode())}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")


class SFTAdapterTests(unittest.TestCase):
    def test_qwen_supervised_suffix_preserves_literal_target(self):
        record = {"instruction": "Rewrite.", "input": "Example", "system": "Assistant", "output": "<think>reason</think><answer>text</answer>"}
        prefix, full = tokenize_sft_record(record, TinyTokenizer(), model_family="qwen")
        self.assertEqual(full[:len(prefix)], prefix)
        self.assertEqual(bytes(full[len(prefix):]).decode(), record["output"] + "<|im_end|>\n")

    def test_ministral_supervised_suffix_preserves_native_target(self):
        record = {"instruction": "Example", "output": "[THINK]reason[/THINK]text"}
        prefix, full = tokenize_sft_record(record, TinyTokenizer(), model_family="ministral")
        self.assertEqual(bytes(full[len(prefix):]).decode(), record["output"])

    def test_native_template_mismatch_fails_instead_of_shifting_loss_mask(self):
        tokenizer = TinyTokenizer()
        tokenizer.apply_chat_template = Mock(side_effect=["wrong prefix", "<assistant>target"])
        with self.assertRaisesRegex(ValueError, "strict prefix"):
            tokenize_sft_record({"instruction": "Example", "output": "target"}, tokenizer, model_family="ministral")

    def test_unknown_family_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported model family"):
            tokenize_sft_record({}, TinyTokenizer(), model_family="unknown")

    def test_native_index_detects_independent_output_head(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
                "language_model.lm_head.weight": "model.safetensors",
                "language_model.model.embed_tokens.weight": "model.safetensors",
            }}))
            self.assertTrue(_mistral3_checkpoint_has_separate_lm_head(root))

    def test_loader_preserves_independent_head_configuration(self):
        config = SimpleNamespace(model_type="mistral3", tie_word_embeddings=True, text_config=SimpleNamespace(use_cache=True))
        model = SimpleNamespace(config=config)
        with (
            patch("magic_model_support.sft.AutoTokenizer.from_pretrained", return_value=SimpleNamespace(pad_token_id=0)),
            patch("magic_model_support.sft.AutoConfig.from_pretrained", return_value=config),
            patch("magic_model_support.sft._mistral3_checkpoint_has_separate_lm_head", return_value=True),
            patch("transformers.Mistral3ForConditionalGeneration.from_pretrained", return_value=model) as load,
        ):
            result, _, _ = load_ministral_model("unused", 0)
        self.assertIs(result, model)
        self.assertFalse(load.call_args.kwargs["config"].tie_word_embeddings)
        self.assertFalse(model.config.text_config.tie_word_embeddings)
        self.assertFalse(model.config.text_config.use_cache)
