import torch
import pandas as pd
from datasets import load_dataset
from tokenizers import Tokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from therapml.lm import TransformerLM
from therapml.train_model.generator import generate_dummy_weights

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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

custom_tokenizer = Tokenizer.from_file("therapml/mmlu/tokenizers/my_bpe_tokenizer.json")

print("Loading GPT-2 baseline...")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
gpt2_model.eval()

def evaluate_custom_model(model, tokenizer, dataset):
    model.eval()
    correct, total = 0, 0
    
    target_chars = ['0', '1', '2', '3']
    target_ids = []
    for char in target_chars:
        tid = tokenizer.token_to_id(char)
        if tid is None:
            tid = tokenizer.token_to_id(f" {char}")
        target_ids.append(tid)

    pad_id = tokenizer.token_to_id("[PAD]")

    with torch.no_grad():
        for row in dataset:
            prompt = str(row['question']).replace('"', '').strip()
            text = f"Question: {prompt} (A) {row['choices'][0]} (B) {row['choices'][1]} (C) {row['choices'][2]} (D) {row['choices'][3]} Answer:"
            
            label_idx = int(row['answer'])
            
            enc = tokenizer.encode(text)
            ids = enc.ids
            
            if len(ids) < MAX_LEN:
                ids = ids + [pad_id] * (MAX_LEN - len(ids))
            elif len(ids) > MAX_LEN:
                ids = ids[:MAX_LEN]
                
            input_ids = torch.tensor([ids]).to(DEVICE)
            
            logits = model(input_ids)
            
            last_token_logits = logits[0, -1, :] 
            
            valid_logits = last_token_logits[target_ids]
            predicted_idx = torch.argmax(valid_logits).item()
            
            if predicted_idx == label_idx:
                correct += 1
            total += 1
            
    return (correct / total) * 100 if total > 0 else 0


def evaluate_gpt2(model, tokenizer, dataset):
    model.eval()
    correct, total = 0, 0
    
    target_ids = [
        tokenizer.encode(" A")[0], 
        tokenizer.encode(" B")[0], 
        tokenizer.encode(" C")[0], 
        tokenizer.encode(" D")[0]
    ]

    with torch.no_grad():
        for row in dataset:
            prompt = str(row['question']).replace('"', '').strip()
            text = f"Question: {prompt}\n(A) {row['choices'][0]}\n(B) {row['choices'][1]}\n(C) {row['choices'][2]}\n(D) {row['choices'][3]}\nAnswer:"
            
            label_idx = int(row['answer'])
            
            inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
            
            if inputs.input_ids.size(1) > MAX_LEN:
                inputs.input_ids = inputs.input_ids[:, :MAX_LEN]
                inputs.attention_mask = inputs.attention_mask[:, :MAX_LEN]
                
            outputs = model(**inputs)
            last_token_logits = outputs.logits[0, -1, :]
            
            valid_logits = last_token_logits[target_ids]
            predicted_idx = torch.argmax(valid_logits).item()
            
            if predicted_idx == label_idx:
                correct += 1
            total += 1
            
    return (correct / total) * 100 if total > 0 else 0

if __name__ == "__main__":
    disciplines = ['global_facts', 'machine_learning', 'philosophy']
    results = []

    for discipline in disciplines:
        print(f"\n=== Evaluating {discipline} ===")
        
        test_dataset = load_dataset("cais/mmlu", discipline, split='validation')
        print(f"Loaded {len(test_dataset)} samples from the 'validation' split.")

        dummy_weights = generate_dummy_weights(
            config["vocab_size"], 
            d_model=config["d_model"], 
            d_ff=config["d_ff"], 
            num_layers=config["num_layers"]
        )

        custom_model = TransformerLM(
            vocab_size=config["vocab_size"],
            context_length=config["context_length"],
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            d_ff=config["d_ff"],
            num_tokens=config["num_tokens"],
            weights=dummy_weights
        ).to(DEVICE)

        checkpoint_path = f"therapml/mmlu/cross_models/{discipline}/best_model.pt"
        try:
            custom_model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
            print(f"Loaded checkpoint: {checkpoint_path}")
        except FileNotFoundError:
            print(f"Checkpoint not found at {checkpoint_path}. Skipping...")
            continue

        print("Evaluating Custom Model...")
        custom_acc = evaluate_custom_model(custom_model, custom_tokenizer, test_dataset)
        
        print("Evaluating GPT-2...")
        gpt2_acc = evaluate_gpt2(gpt2_model, gpt2_tokenizer, test_dataset)
        
        results.append({
            "Discipline": discipline,
            "Custom Model Acc (%)": f"{custom_acc:.2f}",
            "GPT-2 Baseline Acc (%)": f"{gpt2_acc:.2f}"
        })

    print("\n\nFINAL COMPARISON RESULTS")
    df_results = pd.DataFrame(results)
    print(df_results.to_markdown(index=False))