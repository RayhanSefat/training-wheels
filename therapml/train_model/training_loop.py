import torch
import torch.nn as nn
from tokenizers import Tokenizer, decoders
from .generator import generate_dummy_weights
from therapml.lm import RoPE, TransformerLM
from therapml.optimizers import AdamW
from therapml.loss import CrossEntropyLoss
from therapml.misc import clip_grad_norm, get_cosine_schedule_with_warmup
from .common import train_dataset, valid_dataset
from .common import CHECKPOINT_FOLDER
from .common import block_size, batch_size, d_model, num_layers, num_heads, d_ff

import os
os.chdir(os.getcwd())

from pathlib import Path
import re
import glob

device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 50
max_iters = 40000
warmup_stesp = int(0.1 * max_iters)

BASE_DIR = Path(__file__).resolve().parent

tokenizer = Tokenizer.from_file("therapml/train_model/tokenizers/my_bpe_tokenizer.json")
tokenizer.decoder = decoders.ByteLevel()

train_ids = tokenizer.encode(train_dataset["text"][0]).ids
val_ids = tokenizer.encode(valid_dataset["text"][0]).ids

def get_tokens(dataset, num_samples=1000):
    all_ids = []
    for i in range(min(num_samples, len(dataset))):
        text = dataset[i]["text"]
        all_ids.extend(tokenizer.encode(text).ids)
    return torch.tensor(all_ids, dtype=torch.long)

print("Tokenizing...")
train_tokens = get_tokens(train_dataset, num_samples=100000)
val_tokens = get_tokens(valid_dataset, num_samples=10000)
print(f"Number of train tokens = {len(train_tokens)}")

learning_rate = 5e-4

vocab_size = tokenizer.get_vocab_size()
dummy_weights = generate_dummy_weights(vocab_size, d_model=d_model, d_ff=d_ff, num_layers=num_layers)

rope_module = RoPE(
    embedding_dim=d_model // num_heads,
    theta=10000.0,
    context_len=block_size,
    token_positions=torch.arange(block_size).to(device)
)

model = TransformerLM(
    vocab_size=vocab_size,
    context_length=block_size,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    d_ff=d_ff,
    num_tokens=block_size,
    rope=rope_module,
    weights=dummy_weights
).to(device)

model.load_state_dict({
    "ln_final_weight": dummy_weights["ln_final.weight"],
    "lm_head_weight": dummy_weights["lm_head.weight"]
}, strict=False)

optimizer = AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=warmup_stesp,
    num_training_steps=max_iters,
    min_lr_ratio=0.01
)

def get_batch(data, batch_size, block_size):
    if len(data) < block_size + 1:
        raise ValueError(f"Data length ({len(data)}) must be at least block_size + 1 ({block_size + 1})")
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

model.train()

@torch.no_grad()
def estimate_validation_loss():
    model.eval()
    xv, yv = get_batch(val_tokens, batch_size, block_size)
    logits = model(xv)
    B, T, C = logits.shape
    v_loss = criterion(logits.view(B*T, C), yv.view(B*T))
    model.train()
    return v_loss

min_loss = 100000.0

checkpoint_dir = CHECKPOINT_FOLDER
os.makedirs(checkpoint_dir, exist_ok=True)

def get_latest_checkpoint(path):
    files = glob.glob(os.path.join(path, 'checkpoint_iter_*.pt'))
    if not files:
        return None

    latest_file = max(files, key=lambda x: int(re.findall(r'\d+', x)[-1]))
    return latest_file

start_iter = 1
latest_checkpoint = get_latest_checkpoint(checkpoint_dir)

if latest_checkpoint:
    print(f"Loading checkpoint: {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_iter = checkpoint['iter'] + 1
    min_loss = checkpoint.get('loss', 100.0) # Optional: sync min_loss
    print(f"Resuming from iteration {start_iter}")
else:
    print("No checkpoint found. Starting from scratch.")

def check_gradients(model):
    print("--- Gradient Flow Check ---")
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print(f"⚠️ {name:.<50} NO GRADIENT")

if __name__ == "__main__":
    for iter in range(start_iter, max_iters + 1):
        xb, yb = get_batch(train_tokens, batch_size, block_size)
        
        logits = model(xb)
        B, T, C = logits.shape
        loss = criterion(logits.view(B*T, C), yb.view(B*T))
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        validation_loss = estimate_validation_loss()
        check_gradients(model)

        # Save Best Model
        if validation_loss < min_loss:
            torch.save({
                'iter': iter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': validation_loss.item(),
            }, os.path.join(checkpoint_dir, 'checkpoint_best.pt'))
            min_loss = validation_loss
            print("Best model updated")

        # Save Periodic Checkpoint
        if iter % eval_interval == 0:
            torch.save({
                'iter': iter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, os.path.join(checkpoint_dir, f'checkpoint_iter_{iter}.pt'))

        print(f"Step {iter}: Training loss {loss.item():.4f} and Validation loss {validation_loss:.4f}")