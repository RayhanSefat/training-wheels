import torch

def generate_dummy_weights(vocab_size, d_model, d_ff, num_layers):
    weights = {}
    
    weights["token_embeddings.weight"] = torch.randn(vocab_size, d_model)
    weights["ln_final.weight"] = torch.ones(d_model)
    weights["lm_head.weight"] = torch.randn(vocab_size, d_model)

    for i in range(num_layers):
        weights[f"layers.{i}.attn.q_proj.weight"] = torch.randn(d_model, d_model)
        weights[f"layers.{i}.attn.k_proj.weight"] = torch.randn(d_model, d_model)
        weights[f"layers.{i}.attn.v_proj.weight"] = torch.randn(d_model, d_model)
        weights[f"layers.{i}.attn.output_proj.weight"] = torch.randn(d_model, d_model)
        
        weights[f"layers.{i}.ffn.w1.weight"] = torch.randn(d_ff, d_model)
        weights[f"layers.{i}.ffn.w2.weight"] = torch.randn(d_model, d_ff)
        weights[f"layers.{i}.ffn.w3.weight"] = torch.randn(d_ff, d_model)
        
        weights[f"layers.{i}.ln1.weight"] = torch.ones(d_model)
        weights[f"layers.{i}.ln2.weight"] = torch.ones(d_model)
        
    return weights