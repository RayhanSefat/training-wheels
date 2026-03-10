import torch
from tokenizers import Tokenizer
from therapml.lm import RoPE, TransformerLM
from therapml.nn_blocks import softmax
from .generator import generate_dummy_weights
from pathlib import Path
import re
from datasets import load_dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = 'therapml/train_model/models_4/checkpoint_best.pt'
TOKENIZER_PATH = "therapml/train_model/tokenizers/my_tokenizer.json"

dataset = load_dataset("roneneldan/TinyStories")

block_size = 128
d_model = 128
num_layers = 8
num_heads = 4
d_ff = 512

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
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

if Path(CHECKPOINT_PATH).exists():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Successfully loaded checkpoint from iteration {checkpoint['iter']}")
else:
    print("Checkpoint not found. Please check your path.")
    exit()

def evaluate_test_loss():
    test_data = dataset["test"]["text"][0]
    
    test_ids = tokenizer.encode(test_data).ids
    test_tokens = torch.tensor(test_ids, dtype=torch.long).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    losses = []
    
    print(f"Evaluating {len(test_tokens)} tokens from test.csv...")
    
    with torch.no_grad():
        for i in range(0, len(test_tokens) - block_size, block_size):
            x = test_tokens[i:i+block_size].unsqueeze(0)
            y = test_tokens[i+1:i+block_size+1].unsqueeze(0)
            
            logits = model(x)
            B, T, C = logits.shape
            loss = criterion(logits.view(B*T, C), y.view(B*T))
            losses.append(loss.item())
            
    avg_loss = sum(losses) / len(losses) if losses else 0
    return avg_loss

def generate(prompt, max_new_tokens=100, temperature=0.7, top_k=20):
    input_ids = tokenizer.encode(prompt).ids
    idx = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        temp_token = torch.tensor([[3]], device=device)
        while len(idx_cond[0]) < block_size:
            idx_cond = torch.cat((temp_token, idx_cond), dim=1)
        
        with torch.no_grad():
            logits = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, idx_next), dim=1)
            
    decoded_text = tokenizer.decode(idx[0].tolist())

    return re.sub(r'([\"])+\s', r'\0', re.sub(r'\s+([.,!?;:\'])', r'\1', decoded_text))



if __name__ == "__main__":
    # test_loss = evaluate_test_loss()
    # print(f"\n[Test Results]\nAverage Cross Entropy Loss: {test_loss:.4f}")
    
    print("\n--- Model Inference ---")
    prompts = [
        """Ben and Mia like to play in the big room with many books and papers. They see a big box with a lock on it. They want to know what is inside. They look for a key, but they cannot find one. They see a metal thing that looks like a stick. It is a file. They think it can open the lock.

They take the file and try to put it in the lock. They push and pull, but nothing happens. They hear a loud noise. It is a beep. They look at the box. It has a red light and a number. The number is going down. 10, 9, 8...

They do not know what it means. They are scared. They tremble. They want to run away, but they do not know how. The door is locked too. They scream for help, but no one hears them. The number is going down. 3, 2, 1..."""
    ]
    
    for p in prompts:
        print(f"\nPrompt: {p}")
        output = generate(p, max_new_tokens=80)
        print(f"\n\n\n\nGenerated: {output}")