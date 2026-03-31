import torch
import pandas as pd
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_CSV = "therapml/mmlu/dataset/test.csv"
MODEL_NAME = "gpt2"
MAX_LEN = 512

print(f"Loading baseline {MODEL_NAME}...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

choice_tokens = ["A", "B", "C", "D"]
choice_ids = {letter: tokenizer.encode(letter)[0] for letter in choice_tokens}

test_df = pd.read_csv(TEST_CSV)
correct = 0
total = 0

print("Starting Baseline Inference...")

with torch.no_grad():
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        prompt_text = str(row['prompt']).replace('"', '').strip()
        full_text = f"Question: {prompt_text} (A) {row['A']} (B) {row['B']} (C) {row['C']} (D) {row['D']} Answer:"
        ground_truth = str(row['answer']).strip()
        
        inputs = tokenizer(full_text, return_tensors="pt").to(DEVICE)
        input_ids = inputs["input_ids"]
        
        if input_ids.shape[1] > MAX_LEN:
            input_ids = input_ids[:, -MAX_LEN:]
            
        outputs = model(input_ids)
        logits = outputs.logits
        
        last_token_logits = logits[0, -1, :]
        
        candidate_logits = {letter: last_token_logits[idx].item() for letter, idx in choice_ids.items()}
        prediction = max(candidate_logits, key=candidate_logits.get)
        
        if prediction == ground_truth:
            correct += 1
        total += 1

accuracy = (correct / total) * 100
print(f"\n--- GPT-2 (124M) Baseline Results ---")
print(f"Total Questions: {total}")
print(f"Correct Answers: {correct}")
print(f"Accuracy: {accuracy:.2f}%")