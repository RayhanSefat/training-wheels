from typing import Any
from jaxtyping import Float
from therapml.basic_tensor import tensor_multiply, tensor_dot
from therapml.optimizers import SGD, Adam
from torch import Tensor

def run_tensor_multiply(arr1: Float[List, "b x y"], arr2: Float[List, "b y z"]) -> Float[List, "b x z"]:
    return tensor_multiply(arr1, arr2)

def run_tensor_dot(arr1: Float[List, "..."], arr2: Float[List, "..."], dim: int):
    return tensor_dot(arr1, arr2, dim)

def get_sgd_cls() -> Any:
    return SGD

def get_adam_cls() -> Any:
    return Adam

def run_relu(in_features: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    raise NotImplementedError

def run_gelu(in_features: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    raise NotImplementedError

def run_softmax(in_features: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    raise NotImplementedError

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, "d_out d_in"],
    in_features: Float[Tensor, "... d_in"],
) -> Float[Tensor, "... d_out"]:
    raise NotImplementedError

def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    raise NotImplementedError

def run_cross_entropy_loss(
        logits: Float[Tensor, "batch output_dim"],
        ground_truth: Float[Tensor, "batch output_dim"]) -> Float[Tensor, ""]:
    raise NotImplementedError

def run_dropout(input: Float[Tensor, "..."], prob: float) -> Float[Tensor, "..."]:
    raise NotImplementedError

def run_layernorm(input: Float[Tensor, "batch ..."], gamma: Float[Tensor, "batch ..."], beta: Float[Tensor, "batch ..."]) -> Float[Tensor, "batch ..."]:
    raise NotImplementedError

def run_rmsnorm(input: Float[Tensor, "batch ..."], gamma: Float[Tensor, "batch ..."]) -> Float[Tensor, "batch ..."]:
    raise NotImplementedError