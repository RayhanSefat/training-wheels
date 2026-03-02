from .nn_blocks import softmax, Linear, SwiGLU
from .norm import LayerNorm, RMSNorm
from .optimizers import AdamW
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
    def __init__(self, d_model, num_heads, linear_q, linear_k, linear_v, linear_out, mask=None, rope=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.linear_q = linear_q
        self.linear_k = linear_k
        self.linear_v = linear_v
        self.linear_out = linear_out
        self.mask = mask
        self.rope = rope

    def forward(self, in_features):
        batch, seq_len, d_in = in_features.shape
        d_head = self.d_model // self.num_heads

        q = self.linear_q(in_features)
        k = self.linear_k(in_features)
        v = self.linear_v(in_features)

        q = q.view(batch, seq_len, self.num_heads, d_head).transpose(1, 2).transpose(0, 1)
        k = k.view(batch, seq_len, self.num_heads, d_head).transpose(1, 2).transpose(0, 1)
        v = v.view(batch, seq_len, self.num_heads, d_head).transpose(1, 2).transpose(0, 1)

        contexts = [SelfAttention(k[i], v[i], mask=self.mask, rope=self.rope)(q[i]) for i in range(self.num_heads)]
        concatenated = torch.cat(contexts, dim=-1)
        
        return self.linear_out(concatenated)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, ctx_len, multihead_self_attn, swiglu_layer, weights, rope=None):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.ctx_len = ctx_len
        self.multihead_self_attn = multihead_self_attn
        self.swiglu_layer = swiglu_layer
        self.ln1_weight = nn.Parameter(torch.empty_like(weights["ln1.weight"]))
        self.ln2_weight = nn.Parameter(torch.empty_like(weights["ln2.weight"]))
        self.rope = rope

    def forward(self, x):
        ln1_wgt = [[self.ln1_weight for _ in range(x.shape[1])] for _ in range(x.shape[0])]
        ln2_wgt = [[self.ln2_weight for _ in range(x.shape[1])] for _ in range(x.shape[0])]

        norm_x = RMSNorm(ln1_wgt)(x)
        attn_output = self.multihead_self_attn(norm_x)

        h = x + attn_output

        norm_h = RMSNorm(ln2_wgt)(h)
        ffn_output = self.swiglu_layer(norm_h)

        out = h + ffn_output

        return out

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope=None, weights=None):
        super(TransformerLM, self).__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope = rope

        self.q_proj_weights = [weights[f"layers.{i}.attn.q_proj.weight"] for i in range(num_layers)]
        self.k_proj_weights = [weights[f"layers.{i}.attn.k_proj.weight"] for i in range(num_layers)]
        self.v_proj_weights = [weights[f"layers.{i}.attn.v_proj.weight"] for i in range(num_layers)]
        self.o_proj_weights = [weights[f"layers.{i}.attn.output_proj.weight"] for i in range(num_layers)]
        self.ffn_w1_weights = [weights[f"layers.{i}.ffn.w1.weight"] for i in range(num_layers)]
        self.ffn_w2_weights = [weights[f"layers.{i}.ffn.w2.weight"] for i in range(num_layers)]
        self.ffn_w3_weights = [weights[f"layers.{i}.ffn.w3.weight"] for i in range(num_layers)]
        self.ln1_weight = [weights[f"layers.{i}.ln1.weight"] for i in range(num_layers)]
        self.ln2_weight = [weights[f"layers.{i}.ln2.weight"] for i in range(num_layers)]

        self.token_embedding_weight = nn.Parameter(torch.empty((vocab_size, d_model)))
        self.ln_final_weight = nn.Parameter(torch.empty_like(weights["ln_final.weight"]))
        self.lm_head_weight = nn.Parameter(torch.empty_like(weights["lm_head.weight"]))

    def __prepare_linear(self, x_proj_weight):
        out_features, in_features = x_proj_weight.shape
        
        linear = Linear(in_features, out_features) 
        linear.load_state_dict({
            "weight": x_proj_weight
        })

        return linear

    def forward(self, input_indices):
        x = torch.nn.functional.embedding(input_indices, self.token_embedding_weight)

        for i in range(self.num_layers):
            causal_mask = torch.tril(torch.ones(x.shape[1], x.shape[1])).bool()

            multihead_self_attn = MultiHeadSelfAttention(
                self.d_model,
                self.num_heads,
                self.__prepare_linear(self.q_proj_weights[i]),
                self.__prepare_linear(self.k_proj_weights[i]),
                self.__prepare_linear(self.v_proj_weights[i]),
                self.__prepare_linear(self.o_proj_weights[i]),
                mask=causal_mask,
                rope=self.rope
            )

            swiglu_layer = SwiGLU(self.d_model, self.d_ff)
            swiglu_layer.load_state_dict({
                "w1_weight": self.ffn_w1_weights[i],
                "w2_weight": self.ffn_w2_weights[i],
                "w3_weight": self.ffn_w3_weights[i]
            })

            block_weights = {
                "ln1.weight": self.ln1_weight[i],
                "ln2.weight": self.ln2_weight[i]
            }

            block = TransformerBlock(self.d_model, self.num_heads, self.d_ff, self.context_length, multihead_self_attn, swiglu_layer, block_weights, rope=self.rope)
            block.load_state_dict({
                "ln1_weight": self.ln1_weight[i],
                "ln2_weight": self.ln2_weight[i]
            }, strict=False)
            x = block(x)

        ln_final_wgt = [[self.ln_final_weight for _ in range(x.shape[1])] for _ in range(x.shape[0])]
        norm_x = RMSNorm(ln_final_wgt)(x)
        lm_head = Linear(self.d_model, self.vocab_size)
        lm_head.load_state_dict({
            "weight": self.lm_head_weight
        })
        logits = lm_head(norm_x)
        return logits