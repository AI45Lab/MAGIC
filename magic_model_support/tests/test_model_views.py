"""Small checkpoint fixtures exercise the actual conversion entry points."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from magic_model_support.scripts import prepare_qwen35_text_only_view as text_view
from magic_model_support.scripts import prepare_qwen35_vllm_service_view as service_view


class QwenModelViewTests(unittest.TestCase):
    def test_text_and_service_views_load_real_hf_weights(self):
        import torch
        from safetensors.torch import save_file
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

        config = Qwen3_5TextConfig(
            vocab_size=16, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=4, layer_types=["full_attention"], max_position_embeddings=32,
        )
        model = Qwen3_5ForCausalLM(config)
        expected = model.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            reference = root / "reference"
            reference.mkdir()
            save_file(expected, reference / "model.safetensors")
            source.mkdir()
            (source / "config.json").write_text(json.dumps({
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "text_config": config.to_dict(),
            }))
            weights = {
                ("model.language_model." + name[len("model."):] if name.startswith("model.") else name): tensor
                for name, tensor in expected.items()
            }
            weights["model.visual.weight"] = torch.zeros(2, 2)
            keys = list(weights)
            weight_map = {}
            source_bytes = {}
            for i in range(2):
                filename = f"model-{i+1:05d}-of-00002.safetensors"
                shard = {key: weights[key] for key in keys[i::2]}
                save_file(shard, source / filename, metadata={"format": "pt"})
                weight_map.update({key: filename for key in shard})
                source_bytes[filename] = (source / filename).read_bytes()
            (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
            output = root / "text"
            with patch.object(sys, "argv", ["text_view", "--source", str(source), "--reference-text-model", str(reference), "--output", str(output)]):
                text_view.main()
            loaded, info = Qwen3_5ForCausalLM.from_pretrained(output, output_loading_info=True)
            self.assertFalse(info["missing_keys"])
            self.assertFalse(info["unexpected_keys"])
            for name, tensor in expected.items():
                torch.testing.assert_close(loaded.state_dict()[name], tensor, rtol=0, atol=0)
            served = root / "served"
            with patch.object(sys, "argv", ["service_view", "--text-model", str(output), "--multimodal-config-source", str(source), "--output", str(served)]):
                service_view.main()
            served_config = json.loads((served / "config.json").read_text())
            self.assertEqual(served_config["model_type"], "qwen3_5")
            self.assertEqual(served_config["architectures"], ["Qwen3_5ForCausalLM"])
            for filename, content in source_bytes.items():
                self.assertEqual((source / filename).read_bytes(), content)

    def test_qwen_loader_mapping_is_idempotent(self):
        path = Path(__file__).parents[1] / "runtime/qwen/sitecustomize.py"
        spec = importlib.util.spec_from_file_location("qwen_adapter_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict("os.environ", {"MAGIC_QWEN35_FLASH_ATTN_POSIDS_PATCH": "0", "MAGIC_QWEN35_SFT_PREFIX_ADAPTER": "0"}):
            spec.loader.exec_module(module)
        for name in ("model.layers.0.weight", "model.language_model.layers.0.weight", "lm_head.weight"):
            mapped = module._adapt_name(name)
            self.assertEqual(module._adapt_name(mapped), mapped)
