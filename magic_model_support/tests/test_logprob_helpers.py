import unittest

import torch

from magic_model_support.compat.vllm_weight_patch import (
    entropy_from_logits_by_token_blocks,
    logprobs_from_padded_logits_by_sequence,
)


class LogProbabilityHelperTests(unittest.TestCase):
    def test_entropy_values_and_gradients_match_dense_computation(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 5, 7, dtype=torch.float64, requires_grad=True)
        log_probs = logits.log_softmax(-1)
        expected = -(log_probs.exp() * log_probs).sum(-1)
        actual = entropy_from_logits_by_token_blocks(logits, chunk_size=3)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0],
            torch.autograd.grad(expected.sum(), logits)[0],
        )

    def test_ragged_logprobs_values_and_gradients_match_concatenation(self):
        torch.manual_seed(1)
        logits = torch.randn(2, 5, 7, dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0])
        lengths = [3, 5]
        def logprobs_fn(*, logits, labels):
            return logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        actual = logprobs_from_padded_logits_by_sequence(logits, labels, lengths, logprobs_fn)
        expected = logprobs_fn(logits=torch.cat([logits[0, :3], logits[1, :5]]), labels=labels)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0],
            torch.autograd.grad(expected.sum(), logits)[0],
        )
