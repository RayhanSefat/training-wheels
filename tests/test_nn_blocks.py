import numpy as np
import torch
import pytest

from .adapters import run_relu, run_gelu, run_softmax, run_linear, run_swiglu
from .common import FIXTURES_PATH


class TestReLU:
    @pytest.fixture
    def relu_input(self):
        return np.load(FIXTURES_PATH / "relu_input.npy")

    def test_relu_output(self, relu_input):
        tensor = torch.from_numpy(relu_input)
        result = run_relu(tensor)
        result_np = result.detach().numpy() if isinstance(result, torch.Tensor) else result

        # Check shape
        assert result_np.shape == relu_input.shape, f"Expected shape {relu_input.shape}, got {result_np.shape}"

        # Check correctness against PyTorch
        expected = torch.relu(tensor).numpy()
        np.testing.assert_allclose(result_np, expected, rtol=1e-5, atol=1e-8)


class TestGELU:
    @pytest.fixture
    def gelu_input(self):
        return np.load(FIXTURES_PATH / "gelu_input.npy")

    def test_gelu_output(self, gelu_input):
        tensor = torch.from_numpy(gelu_input)
        result = run_gelu(tensor)
        result_np = result.detach().numpy() if isinstance(result, torch.Tensor) else result

        # Check shape
        assert result_np.shape == gelu_input.shape, f"Expected shape {gelu_input.shape}, got {result_np.shape}"

        # Check correctness against PyTorch
        expected = torch.nn.functional.gelu(tensor).numpy()
        np.testing.assert_allclose(result_np, expected, rtol=1e-4, atol=1e-6)


class TestSoftmax:
    @pytest.fixture
    def softmax_input(self):
        return np.load(FIXTURES_PATH / "softmax_input.npy")

    def test_softmax_output(self, softmax_input):
        tensor = torch.from_numpy(softmax_input)
        result = run_softmax(tensor, dim=1)
        result_np = result.detach().numpy() if isinstance(result, torch.Tensor) else result

        # Check shape
        assert result_np.shape == softmax_input.shape, f"Expected shape {softmax_input.shape}, got {result_np.shape}"

        # Check correctness against PyTorch
        expected = torch.nn.functional.softmax(tensor, dim=1).numpy()
        np.testing.assert_allclose(result_np, expected, rtol=1e-5, atol=1e-8)


class TestLinear:
    @pytest.fixture
    def linear_data(self):
        input_data = np.load(FIXTURES_PATH / "linear_input.npy")
        weights = np.load(FIXTURES_PATH / "linear_weights.npy")
        d_out, d_in = weights.shape
        return {
            "input": input_data,
            "weights": weights,
            "d_in": d_in,
            "d_out": d_out,
        }

    def test_linear_output(self, linear_data):
        input_tensor = torch.from_numpy(linear_data["input"])
        weights = torch.from_numpy(linear_data["weights"])

        result = run_linear(
            d_in=linear_data["d_in"], d_out=linear_data["d_out"], weights=weights, in_features=input_tensor
        )
        result_np = result.detach().numpy() if isinstance(result, torch.Tensor) else result

        # Check shape
        expected_shape = (linear_data["input"].shape[0], linear_data["d_out"])
        assert result_np.shape == expected_shape, f"Expected shape {expected_shape}, got {result_np.shape}"

        # Check correctness against PyTorch
        expected = torch.nn.functional.linear(input_tensor, weights, bias=None).numpy()
        np.testing.assert_allclose(result_np, expected, rtol=1e-5, atol=1e-8)


class TestSwiGLU:
    @pytest.fixture
    def swiglu_data(self):
        input_data = np.load(FIXTURES_PATH / "swiglu_input.npy")
        w1 = np.load(FIXTURES_PATH / "swiglu_w1_weight.npy")
        w2 = np.load(FIXTURES_PATH / "swiglu_w2_weight.npy")
        w3 = np.load(FIXTURES_PATH / "swiglu_w3_weight.npy")
        expected = np.load(FIXTURES_PATH / "swiglu_output.npy")
        d_ff, d_model = w1.shape
        return {
            "input": input_data,
            "w1_weight": w1,
            "w2_weight": w2,
            "w3_weight": w3,
            "d_model": d_model,
            "d_ff": d_ff,
            "expected": expected,
        }

    def test_swiglu_output(self, swiglu_data):
        input_tensor = torch.from_numpy(swiglu_data["input"])
        w1 = torch.from_numpy(swiglu_data["w1_weight"])
        w2 = torch.from_numpy(swiglu_data["w2_weight"])
        w3 = torch.from_numpy(swiglu_data["w3_weight"])

        result = run_swiglu(
            d_model=swiglu_data["d_model"],
            d_ff=swiglu_data["d_ff"],
            w1_weight=w1,
            w2_weight=w2,
            w3_weight=w3,
            in_features=input_tensor,
        )
        result_np = result.detach().numpy() if isinstance(result, torch.Tensor) else result

        # Check shape - output should have same shape as input (same d_model)
        assert result_np.shape == swiglu_data["input"].shape, (
            f"Expected shape {swiglu_data['input'].shape}, got {result_np.shape}"
        )

        # Check output contains correct values
        assert np.all(np.isfinite(result_np)), "SwiGLU output should contain only finite values"

        # Check against manually calculated expected output
        np.testing.assert_allclose(
            result_np,
            swiglu_data["expected"],
            rtol=1e-5,
            atol=1e-8,
            err_msg="SwiGLU output does not match expected calculated values",
        )
