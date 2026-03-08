from typing import Any
from jaxtyping import Float, Int, Bool
from therapml.basic_tensor import tensor_multiply, tensor_dot
from therapml.optimizers import SGD, AdamW
from therapml.nn_blocks import ReLU, GELU, softmax, Linear, SwiGLU
from therapml.loss import CrossEntropyLoss
from therapml.dropout import Dropout
from therapml.norm import LayerNorm, RMSNorm
from therapml.lm import RoPE, SelfAttention, MultiHeadSelfAttention, TransformerBlock, TransformerLM
from torch import Tensor
import torch
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir) 

import tensor_lib

def run_tensor_multiply(arr1: Float[List, "b x y"], arr2: Float[List, "b y z"]) -> Float[List, "b x z"]:
    return tensor_lib.tensor_multiply(arr1, arr2)

def run_tensor_dot(arr1: Float[List, "..."], arr2: Float[List, "..."], dim: int):
    return tensor_dot(arr1, arr2, dim)

def get_sgd_cls() -> Any:
    return SGD


def get_adam_cls() -> Any:
    return AdamW


def run_relu(in_features: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    return ReLU(in_features)


def run_gelu(in_features: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    return GELU(in_features)


def run_softmax(in_features: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    return softmax(in_features, dim)


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, "d_out d_in"],
    in_features: Float[Tensor, "... d_in"],
) -> Float[Tensor, "... d_out"]:
    lineear_layer = Linear(d_in, d_out)
    lineear_layer.load_state_dict({"weight": weights.to('cpu')})
    return lineear_layer(in_features)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    swiglu_layer = SwiGLU(d_model, d_ff)
    swiglu_layer.load_state_dict({
        "w1_weight": w1_weight,
        "w2_weight": w2_weight,
        "w3_weight": w3_weight
    })
    return swiglu_layer(in_features)


def run_cross_entropy_loss(
        logits: Float[Tensor, "batch output_dim"],
        ground_truth: Float[Tensor, "batch output_dim"]) -> Float[Tensor, ""]:
    cross_entropy_loss_fn = CrossEntropyLoss()
    return cross_entropy_loss_fn(logits, ground_truth)

def run_dropout(input: Float[Tensor, "..."], prob: float) -> Float[Tensor, "..."]:
    dropout_layer = Dropout(prob)
    return dropout_layer(input)

def run_layernorm(input: Float[Tensor, "batch ..."], gamma: Float[Tensor, "batch ..."], beta: Float[Tensor, "batch ..."]) -> Float[Tensor, "batch ..."]:
    layernorm_layer = LayerNorm(gamma, beta)
    return layernorm_layer(input)

def run_rmsnorm(input: Float[Tensor, "batch ..."], gamma: Float[Tensor, "batch ..."]) -> Float[Tensor, "batch ..."]:
    rmsnorm_layer = RMSNorm(gamma)
    rmsnorm_layer.load_state_dict({
        "gamma": gamma
    })
    return rmsnorm_layer(input)

def run_rope(
    embedding_dim: int,
    theta: float,
    context_len: int,
    input_embeddings: Float[Tensor, "batch ctx_len embedding_dim"],
    token_positions: Int[Tensor, "batch ctx_len"],
) -> Float[Tensor, "batch ctx_len embedding_dim"]:
    return RoPE(embedding_dim, theta, context_len, token_positions)(input_embeddings)

def run_self_attention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... batch keys d_k"],
    V: Float[Tensor, "... batch values d_v"],
    mask: Bool[Tensor, "... batch queries keys"] | None = None,
) -> Float[Tensor, "... batch queries d_v"]:
    """
    Note the number of dimensions here can be greater than 3
    """
    return SelfAttention(K, V, mask)(Q)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, "d_k d_in"],
    k_proj_weight: Float[Tensor, "d_k d_in"],
    v_proj_weight: Float[Tensor, "d_v d_in"],
    o_proj_weight: Float[Tensor, "d_model d_v"],
    in_features: Float[Tensor, "batch ctx_len d_in"],
) -> Float[Tensor, "batch ctx_len d_out"]:
    causal_mask = torch.tril(torch.ones(in_features.shape[1], in_features.shape[1])).bool()

    d_k = q_proj_weight.shape[0]
    d_v = v_proj_weight.shape[0]
    d_in = q_proj_weight.shape[1]

    linear_q = Linear(d_k, d_in)
    linear_q.load_state_dict({
        "weight": q_proj_weight
    })

    linear_k = Linear(d_k, d_in)
    linear_k.load_state_dict({
        "weight": k_proj_weight
    })

    linear_v = Linear(d_v, d_in)
    linear_v.load_state_dict({
        "weight": v_proj_weight
    })

    linear_out = Linear(d_model, d_v)
    linear_out.load_state_dict({
        "weight": o_proj_weight
    })

    multihead_attn_layer = MultiHeadSelfAttention(d_model, num_heads, linear_q, linear_k, linear_v, linear_out, mask=causal_mask)
    return multihead_attn_layer(in_features)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    ctx_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, "d_k d_in"],
    k_proj_weight: Float[Tensor, "d_k d_in"],
    v_proj_weight: Float[Tensor, "d_v d_in"],
    o_proj_weight: Float[Tensor, "d_model d_v"],
    in_features: Float[Tensor, "batch ctx_len d_in"],
    token_positions: Int[Tensor, "batch ctx_len"],
) -> Float[Tensor, "batch ctx_len d_out"]:
    d_head = d_model // num_heads
    rope = RoPE(d_head, theta, ctx_len, token_positions)
    causal_mask = torch.tril(torch.ones(in_features.shape[1], in_features.shape[1])).bool()
    d_k = q_proj_weight.shape[0]
    d_v = v_proj_weight.shape[0]
    d_in = q_proj_weight.shape[1]
    
    linear_q = Linear(d_k, d_in)
    linear_q.load_state_dict({
        "weight": q_proj_weight
    })

    linear_k = Linear(d_k, d_in)
    linear_k.load_state_dict({
        "weight": k_proj_weight
    })

    linear_v = Linear(d_v, d_in)
    linear_v.load_state_dict({
        "weight": v_proj_weight
    })

    linear_out = Linear(d_model, d_v)
    linear_out.load_state_dict({
        "weight": o_proj_weight
    })

    multihead_attn_layer = MultiHeadSelfAttention(d_model, num_heads, linear_q, linear_k, linear_v, linear_out, mask=causal_mask, rope=rope)
    return multihead_attn_layer(in_features)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    ctx_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Note this function should use RoPE.

    Args:
        d_model (int): The dimensionality of the Transformer block input.

        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.

        d_ff (int): Dimensionality of the feed-forward inner layer.

        ctx_len (int): Maximum sequence length of the input_tensor.

        theta (float): RoPE parameter.

        weights (dict[str, Tensor]):
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).

        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    actual_seq_len = in_features.shape[1]

    causal_mask = torch.tril(torch.ones(in_features.shape[1], in_features.shape[1])).bool()
    rope = RoPE(d_model // num_heads, theta, ctx_len, torch.arange(actual_seq_len, device=in_features.device))

    linear_q = Linear(d_model, d_model)
    linear_q.load_state_dict({
        "weight": weights["attn.q_proj.weight"]
    })

    linear_k = Linear(d_model, d_model)
    linear_k.load_state_dict({
        "weight": weights["attn.k_proj.weight"]
    })

    linear_v = Linear(d_model, d_model)
    linear_v.load_state_dict({
        "weight": weights["attn.v_proj.weight"]
    })

    linear_out = Linear(d_model, d_model)
    linear_out.load_state_dict({
        "weight": weights["attn.output_proj.weight"]
    })

    multihead_self_attn = MultiHeadSelfAttention(d_model, num_heads, linear_q, linear_k, linear_v, linear_out, mask=causal_mask, rope=rope)

    swiglu_layer = SwiGLU(d_model, d_ff)
    swiglu_layer.load_state_dict({
        "w1_weight": weights["ffn.w1.weight"],
        "w2_weight": weights["ffn.w2.weight"],
        "w3_weight": weights["ffn.w3.weight"]
    })
    
    transformer_block = TransformerBlock(d_model, num_heads, d_ff, ctx_len, multihead_self_attn, swiglu_layer, weights, rope=rope)
    transformer_block.load_state_dict({
        "ln1_weight": weights["ln1.weight"],
        "ln2_weight": weights["ln2.weight"]
    }, strict=False)

    return transformer_block(in_features)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """
    Similar to before it uses RoPE
    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.

        context_length (int): The maximum number of tokens to process at once.

        d_model (int): The dimensionality of the model embeddings and sublayer outputs.

        num_layers (int): The number of Transformer layers to use.

        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.

        d_ff (int): Dimensionality of the feed-forward inner layer (See Vaswani et al).

        rope_theta (float): The RoPE `theta` parameter.

        weights (dict[str, Tensor]):
            {num_layers} refers to an integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.out_features
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    actual_seq_len = in_indices.shape[1]
    rope = RoPE(d_model // num_heads, rope_theta, context_length, torch.arange(actual_seq_len, device=in_indices.device))
    
    transformer_lm = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, in_indices.shape[1], rope=rope, weights=weights)
    transformer_lm.load_state_dict({
        "token_embedding_weight.weight": weights["token_embeddings.weight"],
        "ln_final_weight": weights["ln_final.weight"],
        "lm_head_weight": weights["lm_head.weight"]
    }, strict=False)

    return transformer_lm(in_indices)
