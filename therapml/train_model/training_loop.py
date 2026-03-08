import torch
import torch.nn as nn
from tokenizers import Tokenizer
from .generator import generate_dummy_weights
from therapml.lm import RoPE, TransformerLM
from therapml.optimizers import AdamW
from therapml.loss import CrossEntropyLoss
from therapml.misc import clip_grad_norm

import os
os.chdir(os.getcwd())

from pathlib import Path
import re
import glob

device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 20
max_iters = 10000

BASE_DIR = Path(__file__).resolve().parent

train_path = BASE_DIR / 'dataset' / 'train.csv'
validation_path = BASE_DIR / 'dataset' / 'validation.csv'
test_path = BASE_DIR / 'dataset' / 'test.csv'

with open(train_path, "r") as f:
    train_data = f.read()
with open(validation_path, "r") as f:
    validation_data = f.read()
with open(test_path, "r") as f:
    test_data = f.read()

tokenizer = Tokenizer.from_file("therapml/train_model/tokenizers/my_tokenizer.json")

dataset = [train_data, validation_data]
encoded_outputs = tokenizer.encode_batch(dataset)

train_ids = tokenizer.encode(train_data).ids
val_ids = tokenizer.encode(validation_data).ids

train_tokens = torch.tensor(train_ids, dtype=torch.long)
val_tokens = torch.tensor(val_ids, dtype=torch.long)

block_size = 256
batch_size = 128
d_model = 128
num_layers = 4
num_heads = 4
d_ff = 512
learning_rate = 5e-4

vocab_size = tokenizer.get_vocab_size()
dummy_weights = generate_dummy_weights(vocab_size, d_model=128, d_ff=512, num_layers=4)

rope_module = RoPE(
    embedding_dim=d_model // num_heads,
    theta=10000.0,
    context_len=block_size,
    token_positions=torch.arange(block_size).to(device)
)

model = TransformerLM(
    vocab_size=vocab_size,
    context_length=block_size,
    d_model=128,
    num_layers=4,
    num_heads=4,
    d_ff=512,
    num_tokens=block_size,
    rope=rope_module,
    weights=dummy_weights
).to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

def get_batch(data, batch_size, block_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
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

min_loss = 100.0

checkpoint_dir = 'therapml/train_model/models/'
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

        validation_loss = estimate_validation_loss()

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