import torch
import pandas as pd
from tqdm import tqdm
from therapml.lm import TransformerLM
from tokenizers import Tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "therapml/mmlu/models/best_model.pt"
TEST_CSV = "therapml/mmlu/dataset/test.csv"
TOKENIZER_PATH = "therapml/mmlu/tokenizers/my_bpe_tokenizer.json"
MAX_LEN = 512

config = {
    "vocab_size": 64000,
    "context_length": 512,
    "d_model": 768,
    "num_layers": 12,
    "num_heads": 12,
    "d_ff": 3072,
    "num_tokens": 512
}

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

from therapml.train_model.generator import generate_dummy_weights
model = TransformerLM(**config, weights=generate_dummy_weights(config["vocab_size"], d_model=config["d_model"], d_ff=config["d_ff"], num_layers=config["num_layers"])).to(DEVICE)

print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

test_df = pd.read_csv(TEST_CSV)
correct = 0
total = 0

print("Starting Inference on Test Set...")

with torch.no_grad():
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        prompt = str(row['prompt']).replace('"', '').strip()
        text = f"Question: {prompt} (A) {row['A']} (B) {row['B']} (C) {row['C']} (D) {row['D']} Answer:"
        ground_truth = str(row['answer']).strip()
        
        full_tokens = tokenizer.encode(text).ids
        
        if len(full_tokens) > MAX_LEN:
            tokens = full_tokens[-MAX_LEN:]
        else:
            tokens = full_tokens + [tokenizer.token_to_id("[PAD]")] * (MAX_LEN - len(full_tokens))
            
        input_tensor = torch.tensor([tokens]).to(DEVICE)
        
        logits = model(input_tensor)
        
        actual_len = min(len(full_tokens), MAX_LEN)
        last_token_logits = logits[0, actual_len - 1, :]
        
        choice_ids = {
            "A": tokenizer.token_to_id("A"),
            "B": tokenizer.token_to_id("B"),
            "C": tokenizer.token_to_id("C"),
            "D": tokenizer.token_to_id("D")
        }
        
        probs = {choice: last_token_logits[idx].item() for choice, idx in choice_ids.items()}
        prediction = max(probs, key=probs.get)
        
        if prediction == ground_truth:
            correct += 1
        total += 1

accuracy = (correct / total) * 100
print(f"\n--- MMLU Results ---")
print(f"Total Questions: {total}")
print(f"Correct Answers: {correct}")
print(f"Accuracy: {accuracy:.2f}%")