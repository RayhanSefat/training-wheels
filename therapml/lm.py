import numpy as np
from .nn_blocks import softmax, Linear
import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, embedding_dim, theta, context_len, token_positions):
        super(RoPE, self).__init__()
        self.embedding_dim = embedding_dim
        self.theta = theta
        self.context_len = context_len
        self.token_positions = token_positions

    def forward(self, input_embeddings):
        dim_indices = np.arange(0, self.embedding_dim, 2)
        omega = 1.0 / (self.theta ** (dim_indices / self.embedding_dim))
        
        m = self.token_positions.reshape(-1, 1)
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

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.q_weight = q_proj_weight
        self.k_weight = k_proj_weight
        self.v_weight = v_proj_weight
        self.o_weight = o_proj_weight

    def forward(self, in_features):
        batch, seq_len, d_in = in_features.shape
        d_head = self.d_model // self.num_heads

        q = Linear(0, 0, self.q_weight)(in_features)
        k = Linear(0, 0, self.k_weight)(in_features)
        v = Linear(0, 0, self.v_weight)(in_features)

        q = q.view(batch, seq_len, self.num_heads, d_head).transpose(1, 2).transpose(0, 1)
        k = k.view(batch, seq_len, self.num_heads, d_head).transpose(1, 2).transpose(0, 1)
        v = v.view(batch, seq_len, self.num_heads, d_head).transpose(1, 2).transpose(0, 1)

        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()

        contexts = [SelfAttention(k[i], v[i], mask=causal_mask)(q[i]) for i in range(self.num_heads)]
        concatenated = torch.cat(contexts, dim=-1)
        return Linear(0, 0, self.o_weight)(concatenated)