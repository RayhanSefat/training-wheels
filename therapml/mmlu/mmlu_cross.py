from datasets import load_dataset
from tokenizers import Tokenizer, decoders
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from therapml.lm import TransformerLM
from therapml.train_model.generator import generate_dummy_weights
import random
import os

random.seed(42)
torch.manual_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20
SAVE_INTERVAL = 10
VALIDATION_INTERVAL = 1
BATCH_SIZE = 8
MAX_LEN = 512
LR = 3e-4

tokenizer = Tokenizer.from_file("therapml/mmlu/tokenizers/my_bpe_tokenizer.json")
tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"))
tokenizer.decoder = decoders.ByteLevel()

class MMLUDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len):
        df = pd.DataFrame(dataset)
        self.data = []
        target_col = 'answer' 

        for i, row in df.iterrows():
            prompt = str(row['question']).replace('"', '').strip()
            text = f"Question: {prompt} (A) {row['choices'][0]} (B) {row['choices'][1]} (C) {row['choices'][2]} (D) {row['choices'][3]} Answer:"
            target_char = str(row[target_col]).strip()
            
            enc = tokenizer.encode(text)
            target_id = tokenizer.token_to_id(target_char)
            
            if target_id is None:
                target_id = tokenizer.token_to_id(f" {target_char}")

            if target_id is not None and len(enc.ids) < max_len:
                self.data.append({
                    "input_ids": torch.tensor(enc.ids),
                    "target_id": torch.tensor(target_id)
                })
        
        print(f"Successfully loaded {len(self.data)} samples into dataset.")
        if len(self.data) == 0:
            raise ValueError("Dataset is empty! Check if target IDs are None or if max_len is too small.")

    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]["input_ids"], self.data[idx]["target_id"]
    
def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    targets = torch.stack([item[1] for item in batch])
    
    padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, 
        batch_first=True, 
        padding_value=tokenizer.token_to_id("[PAD]")
    )
    
    curr_batch_size, curr_seq_len = padded.size()
    
    if curr_seq_len < MAX_LEN:
        padding = torch.full(
            (curr_batch_size, MAX_LEN - curr_seq_len), 
            tokenizer.token_to_id("[PAD]"), 
            dtype=torch.long
        )
        padded = torch.cat([padded, padding], dim=1)
    elif curr_seq_len > MAX_LEN:
        padded = padded[:, :MAX_LEN]

    return padded.to(DEVICE), targets.to(DEVICE)

config = {
    "vocab_size": 64000,
    "context_length": 512,
    "d_model": 768,
    "num_layers": 12,
    "num_heads": 12,
    "d_ff": 3072,
    "num_tokens": 512
}

def load_and_train_mmlu(discipline):
    train_dataset = load_dataset("cais/mmlu", discipline, split='test')
    validation_dataset = load_dataset("cais/mmlu", discipline, split='validation')
    train_ds = MMLUDataset(train_dataset, tokenizer, MAX_LEN)
    valid_ds = MMLUDataset(validation_dataset, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    
    
    weights = generate_dummy_weights(config["vocab_size"], d_model=config["d_model"], d_ff=config["d_ff"], num_layers=config["num_layers"])

    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        num_tokens=config["num_tokens"],
        weights=weights
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float('inf')

    checkpoints = f"therapml/mmlu/cross_models/{discipline}"
    os.makedirs(checkpoints, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0
        
        sample_indices = random.sample(range(len(train_ds)), min(128, len(train_ds)))
        sample_ds = torch.utils.data.Subset(train_ds, sample_indices)
        sample_loader = DataLoader(sample_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

        
        for x, y in sample_loader:
            optimizer.zero_grad()
            logits = model(x) 
            
            last_token_logits = logits[:, -1, :] 
            
            loss = criterion(last_token_logits, y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        if epoch % VALIDATION_INTERVAL == 0:
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for vx, vy in valid_loader:
                    v_logits = model(vx)[:, -1, :]
                    v_loss = criterion(v_logits, vy)
                    total_val_loss += v_loss.item()
            
            avg_val_loss = total_val_loss / len(valid_loader)
            print(f"Epoch {epoch}: Train Loss {total_train_loss/len(train_loader):.4f} | Val Loss {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f"{checkpoints}/best_model.pt")
                print("⭐ New best model saved!")

        if epoch % SAVE_INTERVAL == 0:
            torch.save(model.state_dict(), f"{checkpoints}/model_epoch_{epoch}.pt")

    print(f"Training complete for {discipline}!")

if __name__ == "__main__":
    disciplines = ['global_facts', 'machine_learning', 'philosophy']
    
    for discipline in disciplines:
        print(f"\n=== Training on {discipline} ===")
        load_and_train_mmlu(discipline)