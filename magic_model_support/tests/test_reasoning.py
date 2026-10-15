import unittest

from magic_model_support.reasoning import mistral_reasoning_api_messages, parse_native_reasoning_response


class ReasoningAdapterTests(unittest.TestCase):
    def test_system_prompt_uses_structured_thinking_block(self):
        messages = [{"role": "system", "content": "before[THINK]example[/THINK]after"}]
        encoded = mistral_reasoning_api_messages(messages)
        self.assertEqual(encoded[0]["content"], [
            {"type": "text", "text": "before"},
            {"type": "thinking", "thinking": "example", "closed": True},
            {"type": "text", "text": "after"},
        ])
        self.assertIsInstance(messages[0]["content"], str)

    def test_complete_native_output_separates_reasoning_and_answer(self):
        self.assertEqual(parse_native_reasoning_response("[THINK]reason[/THINK]answer"), ("reason", "answer", None))

    def test_incomplete_reasoning_is_not_a_visible_answer(self):
        self.assertEqual(parse_native_reasoning_response("[THINK]unfinished"), ("", "", "missing_close_marker"))

    def test_direct_answer_retains_missing_marker_diagnostic(self):
        self.assertEqual(parse_native_reasoning_response("answer"), ("", "answer", "missing_native_markers"))
