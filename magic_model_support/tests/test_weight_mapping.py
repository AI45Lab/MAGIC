import unittest

from magic_model_support.compat.vllm_weight_patch import (
    adapt_pixtral_text_only_weight_name,
    iter_pixtral_text_only_weights,
)


class WeightMappingTests(unittest.TestCase):
    def test_multimodal_tensors_are_excluded_and_text_keys_are_mapped(self):
        self.assertIsNone(adapt_pixtral_text_only_weight_name("model.vision_tower.patch_conv.weight"))
        self.assertIsNone(adapt_pixtral_text_only_weight_name("model.multi_modal_projector.linear.weight"))
        self.assertEqual(adapt_pixtral_text_only_weight_name("model.language_model.model.layers.0.weight"), "language_model.model.layers.0.weight")
        self.assertEqual(adapt_pixtral_text_only_weight_name("model.language_model.embed_tokens.weight"), "language_model.model.embed_tokens.weight")
        self.assertEqual(adapt_pixtral_text_only_weight_name("lm_head.weight"), "language_model.lm_head.weight")

    def test_ambiguous_head_update_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "ambiguous weight update"):
            list(iter_pixtral_text_only_weights([
                ("lm_head.weight", object()),
                ("language_model.lm_head.weight", object()),
            ]))
