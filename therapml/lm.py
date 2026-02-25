from .nn_blocks import softmax, Linear, SwiGLU
from .norm import LayerNorm, RMSNorm
import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, embedding_dim, theta, context_len, token_positions):
        super(RoPE, self).__init__()
        self.embedding_dim = embedding_dim
        self.theta = theta
        self.token_positions = token_positions

    def forward(self, x):
        dim_indices = torch.arange(0, self.embedding_dim, 2).float()
        omega = 1.0 / (self.theta ** (dim_indices / self.embedding_dim))
        
        m = self.token_positions.reshape(-1, 1).float()
        angles = m * omega # Broadcasting: [seq_len, 1] * [d_head/2] -> [seq_len, d_head/2]
        
        cos = angles.cos()
        sin = angles.sin()

        batch_size, seq_len, _ = x.shape    
        x_reshaped = x.reshape(batch_size, seq_len, -1, 2)
        x1 = x_reshaped[..., 0]
        x2 = x_reshaped[..., 1]
        
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        
        return torch.stack([out1, out2], dim=-1).reshape(batch_size, seq_len, self.embedding_dim).to(torch.float32)

class SelfAttention(nn.Module):
    def __init__(self, k, v, mask=None, rope=None):
        super(SelfAttention, self).__init__()
        self.k = k
        self.v = v
        self.mask = mask
        self.rope = rope

    def forward(self, q):
        d_k = q.shape[-1]

        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(self.k)
        else:
            k = self.k

        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        
        if self.mask is not None:
            scores = scores.masked_fill(self.mask == False, float("-inf"))
        
        attn_probs = torch.softmax(scores, dim=-1)
        return torch.matmul(attn_probs, self.v)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_k, d_v, d_in, mask=None, rope=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.d_in = d_in
        self.q_weight = nn.Parameter(torch.empty(d_k, d_in))
        self.k_weight = nn.Parameter(torch.empty(d_k, d_in))
        self.v_weight = nn.Parameter(torch.empty(d_v, d_in))
        self.o_weight = nn.Parameter(torch.empty(d_model, d_v))
        self.mask = mask
        self.rope = rope

    def __get_qkv(self, in_features):
        linear_obj = Linear(self.d_in, self.d_model)
        linear_obj.load_state_dict({
            "weight": self.q_weight
        })
        q = linear_obj(in_features)
        linear_obj.load_state_dict({
            "weight": self.k_weight
        })
        k = linear_obj(in_features)
        linear_obj.load_state_dict({
            "weight": self.v_weight
        })
        v = linear_obj(in_features)
        return q, k, v

    def forward(self, in_features):
        batch, seq_len, d_in = in_features.shape
        d_head = self.d_model // self.num_heads

        q, k, v = self.__get_qkv(in_features)

        q = q.view(batch, seq_len, self.num_heads, d_head).transpose(1, 2).transpose(0, 1)
        k = k.view(batch, seq_len, self.num_heads, d_head).transpose(1, 2).transpose(0, 1)
        v = v.view(batch, seq_len, self.num_heads, d_head).transpose(1, 2).transpose(0, 1)

        contexts = [SelfAttention(k[i], v[i], mask=self.mask, rope=self.rope)(q[i]) for i in range(self.num_heads)]
        concatenated = torch.cat(contexts, dim=-1)
        
        linear_out = Linear(self.d_model, self.d_model)
        linear_out.load_state_dict({
            "weight": self.o_weight
        })
        return linear_out(concatenated)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, ctx_len, weights, rope=None):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.ctx_len = ctx_len
        self.q_proj_weight = weights["attn.q_proj.weight"]
        self.k_proj_weight = weights["attn.k_proj.weight"]
        self.v_proj_weight = weights["attn.v_proj.weight"]
        self.o_proj_weight = weights["attn.output_proj.weight"]
        self.ln1_weight = weights["ln1.weight"]
        self.ffn_w1_weight = weights["ffn.w1.weight"]
        self.ffn_w2_weight = weights["ffn.w2.weight"]
        self.ffn_w3_weight = weights["ffn.w3.weight"]
        self.ln2_weight = weights["ln2.weight"]
        self.rope = rope

    def __get_attention(self, mask=None):
        multihead_self_attn = MultiHeadSelfAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_k=self.q_proj_weight.shape[0],
            d_v=self.v_proj_weight.shape[0],
            d_in=self.q_proj_weight.shape[1],
            mask=mask,
            rope=self.rope
        )

        multihead_self_attn.load_state_dict({
            "q_weight": self.q_proj_weight,
            "k_weight": self.k_proj_weight,
            "v_weight": self.v_proj_weight,
            "o_weight": self.o_proj_weight
        })
        
        return multihead_self_attn

    def __get_attn_output(self, x):
        seq_len = x.shape[1]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        attention = self.__get_attention(mask=causal_mask)
        return attention(x)

    def __get_ffn_output(self, x):
        swiglu_layer = SwiGLU(self.d_model, self.d_ff)
        swiglu_layer.load_state_dict({
            "w1_weight": self.ffn_w1_weight,
            "w2_weight": self.ffn_w2_weight,
            "w3_weight": self.ffn_w3_weight
        })
        return swiglu_layer(x)

    def forward(self, x):
        ln1_wgt = [[self.ln1_weight for _ in range(x.shape[1])] for _ in range(x.shape[0])]
        ln2_wgt = [[self.ln2_weight for _ in range(x.shape[1])] for _ in range(x.shape[0])]

        norm_x = RMSNorm(ln1_wgt)(x)
        attn_output = self.__get_attn_output(norm_x)

        h = x + attn_output

        norm_h = RMSNorm(ln2_wgt)(h)
        ffn_output = self.__get_ffn_output(norm_h)

        out = h + ffn_output

        return out