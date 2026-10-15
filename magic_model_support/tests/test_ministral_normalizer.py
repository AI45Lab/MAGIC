import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from magic_model_support.compat.normalize_ministral3_merged import normalize


class MinistralNormalizerTests(unittest.TestCase):
    def test_untied_head_is_preserved_and_normalization_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "config.json").write_text(json.dumps({"text_config": {}, "tie_word_embeddings": True}))
            embedding, head = torch.zeros(2, 2), torch.ones(2, 2)
            save_file({"model.language_model.model.embed_tokens.weight": embedding,
                       "language_model.lm_head.weight": head}, model / "model.safetensors")
            normalize(model)
            result = load_file(model / "model.safetensors")
            self.assertTrue(torch.equal(result["lm_head.weight"], head))
            self.assertFalse(torch.equal(result["lm_head.weight"], result["model.language_model.embed_tokens.weight"]))
            self.assertEqual(normalize(model)["status"], "already_normalized")
            config = json.loads((model / "config.json").read_text())
            self.assertFalse(config["tie_word_embeddings"])
            self.assertFalse(config["text_config"]["tie_word_embeddings"])

    def test_missing_or_conflicting_head_fails_without_replacing_weights(self):
        for heads in ({}, {"lm_head.weight": torch.zeros(2, 2), "language_model.lm_head.weight": torch.ones(2, 2)}):
            with self.subTest(heads=list(heads)), tempfile.TemporaryDirectory() as directory:
                model = Path(directory)
                (model / "config.json").write_text('{"text_config": {}}')
                weights = model / "model.safetensors"
                save_file({"model.language_model.model.embed_tokens.weight": torch.zeros(2, 2), **heads}, weights)
                original = weights.read_bytes()
                with self.assertRaises(RuntimeError):
                    normalize(model)
                self.assertEqual(weights.read_bytes(), original)
