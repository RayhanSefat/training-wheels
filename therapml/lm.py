import numpy as np
from .nn_blocks import softmax, Linear
import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, embedding_dim, theta, context_len):
        super(RoPE, self).__init__()
        self.embedding_dim = embedding_dim
        self.theta = theta
        self.context_len = context_len

    def forward(self, input_embeddings, token_positions):
        dim_indices = np.arange(0, self.embedding_dim, 2)
        omega = 1.0 / (self.theta ** (dim_indices / self.embedding_dim))
        
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
        
        return np.stack([out1, out2], axis=-1).reshape(batch_size, seq_len, self.embedding_dim)

class SelfAttention(nn.Module):
    def __init__(self, k, v, mask=None):
        super(SelfAttention, self).__init__()
        self.k = k
        self.v = v
        self.mask = mask

    def forward(self, q):
        d_k = q.shape[-1]
        
        scores = np.matmul(q, self.k.transpose(-2, -1)) / (d_k ** 0.5)
        
        if self.mask is not None:
            scores = scores.masked_fill(self.mask == False, float("-inf"))
        
        attn_probs = softmax(scores, dim=-1)
        return np.matmul(attn_probs, self.v)        

def self_attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    
    scores = np.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    
    if mask is not None:
        scores = scores.masked_fill(mask == False, float("-inf"))
    
    attn_probs = softmax(scores, dim=-1)
    return np.matmul(attn_probs, v)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj_weight = q_proj_weight
        self.k_proj_weight = k_proj_weight
        self.v_proj_weight = v_proj_weight
        self.o_proj_weight = o_proj_weight

    def forward(self, in_features):
        batch_size, ctx_len, d_in = in_features.shape

        q_proj = Linear(0, 0, self.q_proj_weight)(in_features)
        k_proj = Linear(0, 0, self.k_proj_weight)(in_features)
        v_proj = Linear(0, 0, self.v_proj_weight)(in_features)

        d_k = q_proj.shape[-1]
        d_head_k = d_k // self.num_heads
        d_v = v_proj.shape[-1]
        d_head_v = d_v // self.num_heads

        q_proj = q_proj.reshape(batch_size, ctx_len, self.num_heads, d_head_k).transpose(0, 2, 1, 3)
        k_proj = k_proj.reshape(batch_size, ctx_len, self.num_heads, d_head_k).transpose(0, 2, 1, 3)
        v_proj = v_proj.reshape(batch_size, ctx_len, self.num_heads, d_head_v).transpose(0, 2, 1, 3)

        score = np.matmul(q_proj, k_proj.transpose(0, 1, 3, 2)) / (d_head_k ** 0.5)
        attn_weights = softmax(score, -1)

        context = np.matmul(attn_weights, v_proj)
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, ctx_len, d_v)

        output = Linear(0, 0, self.o_proj_weight)(context)
        return output