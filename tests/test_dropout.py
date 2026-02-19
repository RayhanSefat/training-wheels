import torch

from .adapters import run_dropout


class TestDropout:
    def test_dropout_shape(self):
        input_tensor = torch.randn(2, 3, 4, 5)
        prob = 0.5

        output = run_dropout(input_tensor, prob)

        assert output.shape == input_tensor.shape, f"Expected shape {input_tensor.shape}, got {output.shape}"

    def test_dropout_shape_1d(self):
        input_tensor = torch.randn(100)
        prob = 0.3

        output = run_dropout(input_tensor, prob)

        assert output.shape == input_tensor.shape

    def test_dropout_zero_prob(self):
        input_tensor = torch.randn(100)

        output = run_dropout(input_tensor, prob=0.0)

        torch.testing.assert_close(output, input_tensor)

    def test_dropout_high_prob(self):
        input_tensor = torch.randn(1000)
        prob = 1.00

        output = run_dropout(input_tensor, prob)

        # All values should be zero
        torch.testing.assert_close(output, torch.zeros_like(input_tensor))

    def test_dropout_drop_ratio(self):
        input_tensor = torch.ones(10000)  # Use ones to make it easier to count zeros
        probs = [0.1, 0.3, 0.5, 0.7]

        for prob in probs:
            output = run_dropout(input_tensor, prob)

            num_zeros = (output == 0).sum().item()
            total_values = output.numel()
            actual_drop_ratio = num_zeros / total_values

            # Allow some tolerance (within 5% of expected)
            assert abs(actual_drop_ratio - prob) < 0.05, (
                f"For prob={prob}, expected drop ratio ~{prob}, got {actual_drop_ratio}"
            )

    def test_dropout_scaling_of_active_elements(self):
        input_tensor = torch.ones(10000)
        prob = 0.5

        output = run_dropout(input_tensor, prob)

        # Get non-zero elements
        non_zero_mask = output != 0
        active_elements = output[non_zero_mask]

        # Active elements should be scaled by 1/(1-prob)
        expected_scale = 1.0 / (1.0 - prob)

        # All non-zero values should equal the expected scale (within floating point precision)
        torch.testing.assert_close(
            active_elements, torch.full_like(active_elements, expected_scale), rtol=1e-5, atol=1e-5
        )

    def test_dropout_multiple_runs(self):
        input_tensor = torch.randn(5000)
        prob = 0.3
        num_runs = 100

        drop_ratios = []

        for _ in range(num_runs):
            output = run_dropout(input_tensor, prob)
            num_zeros = (output == 0).sum().item()
            total_values = output.numel()
            drop_ratio = num_zeros / total_values
            drop_ratios.append(drop_ratio)

        # Average drop ratio should be close to prob
        avg_drop_ratio = sum(drop_ratios) / len(drop_ratios)

        assert abs(avg_drop_ratio - prob) < 0.02, f"Expected average drop ratio ~{prob}, got {avg_drop_ratio}"

    def test_dropout_scaling_multiple_runs(self):
        input_tensor = torch.ones(10000)
        prob = 0.4
        num_runs = 50
        expected_scale = 1.0 / (1.0 - prob)

        for _ in range(num_runs):
            output = run_dropout(input_tensor, prob)

            # Get non-zero elements
            non_zero_mask = output != 0
            if non_zero_mask.sum() > 0:  # Only check if there are non-zero elements
                active_elements = output[non_zero_mask]

                # All non-zero values should equal expected_scale
                torch.testing.assert_close(
                    active_elements, torch.full_like(active_elements, expected_scale), rtol=1e-5, atol=1e-5
                )

    def test_dropout_preserves_expected_value(self):
        input_tensor = torch.ones(50000)
        prob = 0.5

        output = run_dropout(input_tensor, prob)

        # Expected value should be preserved
        # E[output] = E[input] * (1 - prob) * 1/(1-prob) = E[input]
        expected_mean = input_tensor.mean().item()
        actual_mean = output.mean().item()

        # Allow small tolerance due to sampling variance
        assert abs(actual_mean - expected_mean) < 0.1, f"Expected mean ~{expected_mean}, got {actual_mean}"
