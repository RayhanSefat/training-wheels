import numpy as np
from .nn_blocks import softmax, linear

def rope(embedding_dim, theta, context_len, input_embeddings, token_positions):
    dim_indices = np.arange(0, embedding_dim, 2)
    omega = 1.0 / (theta ** (dim_indices / embedding_dim))
    
    m = token_positions.reshape(-1, 1)
    angles = m * omega
    
    cos = np.cos(angles)
    sin = np.sin(angles)

    batch_size, seq_len, _ = input_embeddings.shape    
    x = input_embeddings.reshape(batch_size, seq_len, -1, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    
    return np.stack([out1, out2], axis=-1).reshape(batch_size, seq_len, embedding_dim)

def self_attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    
    scores = np.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    
    if mask is not None:
        scores = scores.masked_fill(mask == False, float("-inf"))
    
    attn_probs = softmax(scores, dim=-1)
    return np.matmul(attn_probs, v)

def multihead_self_attention(d_model,
        num_heads,
        q_proj_weight,
        k_proj_weight,
        v_proj_weight,
        o_proj_weight,
        in_features
    ):
    batch_size, ctx_len, d_in = in_features.shape

    q_proj = linear(0, 0, q_proj_weight, in_features) # batch_size x ctx_len x d_k
    k_proj = linear(0, 0, k_proj_weight, in_features) # batch_size x ctx_len x d_k
    v_proj = linear(0, 0, v_proj_weight, in_features) # batch_size x ctx_len x d_v

    d_k = q_proj_weight.shape[-2]
    d_head_k = d_k // num_heads
    d_v = v_proj_weight.shape[-2]
    d_head_v = d_v // num_heads

    q_proj = q_proj.reshape(batch_size, ctx_len, num_heads, d_head_k).transpose(0, 2, 1, 3) # batch_size x num_heads x ctx_len x d_head_k
    k_proj = k_proj.reshape(batch_size, ctx_len, num_heads, d_head_k).transpose(0, 2, 1, 3) # batch_size x num_heads x ctx_len x d_head_k
    v_proj = v_proj.reshape(batch_size, ctx_len, num_heads, d_head_v).transpose(0, 2, 1, 3) # batch_size x num_heads x ctx_len x d_head_v

    score = np.matmul(q_proj, k_proj.transpose(0, 1, 3, 2)) / (d_head_k ** 0.5) # batch_size x num_heads x ctx_len x ctx_len
    attn_weights = softmax(score, -1) # batch_size x num_heads x ctx_len x ctx_len

    context = np.matmul(attn_weights, v_proj) # batch_size x num_heads x ctx_len x d_head_v
    context = context.transpose(0, 2, 1, 3).reshape(batch_size, ctx_len, d_v) # batch_size x ctx_len x d_v

    output = linear(0, 0, o_proj_weight, context) # batch_size x ctx_len x d_model
    return output

def multihead_self_attention_with_rope(d_model,
        num_heads,
        ctx_len,
        theta,
        q_proj_weight,
        k_proj_weight,
        v_proj_weight,
        o_proj_weight,
        in_features,
        token_positions
    ):
    raise NotImplementedError