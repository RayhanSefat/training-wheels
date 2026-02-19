import numpy as np
import torch
import pytest

from .adapters import run_cross_entropy_loss
from .common import FIXTURES_PATH


class TestCrossEntropyLoss:
    @pytest.fixture
    def ce_loss_data(self):
        logits = np.load(FIXTURES_PATH / "cross_entropy_logits.npy")
        ground_truth = np.load(FIXTURES_PATH / "cross_entropy_ground_truth.npy")
        return {
            "logits": logits,
            "ground_truth": ground_truth,
        }

    def test_cross_entropy_loss_output(self, ce_loss_data):
        logits_tensor = torch.from_numpy(ce_loss_data["logits"])
        ground_truth_tensor = torch.from_numpy(ce_loss_data["ground_truth"])

        result = run_cross_entropy_loss(logits_tensor, ground_truth_tensor)
        result_np = result.numpy() if isinstance(result, torch.Tensor) else result

        # Result should be a scalar or 1D tensor with batch size
        # Cross entropy typically returns loss per sample or a single scalar
        assert result_np.ndim <= 1, f"Expected scalar or 1D output, got shape {result_np.shape}"

        expected = torch.nn.functional.cross_entropy(logits_tensor, ground_truth_tensor)

        np.testing.assert_allclose(
            result_np, expected.numpy(), atol=1e-4, err_msg="Cross entropy loss doesn't match manual computation"
        )

    def test_cross_entropy_loss_with_large_logits(self, ce_loss_data):
        """Test that adding a large constant to logits doesn't change the loss.

        This tests numerical stability: softmax(x + c) = softmax(x) for any constant c,
        so cross entropy loss should remain invariant.
        """
        logits_tensor = torch.from_numpy(ce_loss_data["logits"])
        ground_truth_tensor = torch.from_numpy(ce_loss_data["ground_truth"])

        # Compute loss with original logits
        result_original = run_cross_entropy_loss(logits_tensor, ground_truth_tensor)
        result_original_np = result_original.numpy() if isinstance(result_original, torch.Tensor) else result_original

        # Add a large constant to all logits
        large_constant = 1e4
        logits_shifted = logits_tensor + large_constant

        # Compute loss with shifted logits
        result_shifted = run_cross_entropy_loss(logits_shifted, ground_truth_tensor)
        result_shifted_np = result_shifted.numpy() if isinstance(result_shifted, torch.Tensor) else result_shifted

        # Results should be identical (within numerical precision)
        np.testing.assert_allclose(
            result_original_np,
            result_shifted_np,
            atol=1e-4,
            err_msg="Cross entropy loss changed when adding constant to logits",
        )
