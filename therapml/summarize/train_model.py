import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from therapml.lm import TransformerLM
from therapml.train_model.generator import generate_dummy_weights
from tokenizers import Tokenizer
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20000
SAVE_INTERVAL = 500
VALIDATION_INTERVAL = 1
BATCH_SIZE = 16
MAX_LEN = 512
LR = 1e-4

tokenizer = Tokenizer.from_file("therapml/summarize/tokenizers/my_bpe_tokenizer.json")
PAD_ID = tokenizer.token_to_id("[PAD]")

class SummarizationDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len):
        df = pd.read_csv(csv_path)
        self.data = []
        
        for _, row in df.iterrows():
            dialogue = str(row['dialogue']).strip()
            if dialogue.startswith('"') and dialogue.endswith('"'):
                dialogue = dialogue[1:-1]
            
            summary = str(row['summary']).strip()
            if summary.startswith('"') and summary.endswith('"'):
                summary = summary[1:-1]
            
            full_text = f"Dialogue: {dialogue} Summary: {summary} [SEP]"
            
            enc = tokenizer.encode(full_text)
            ids = enc.ids
            
            if len(ids) <= max_len and len(ids) > 10:
                self.data.append(torch.tensor(ids))

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def collate_fn(batch):
    padded = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=PAD_ID)
    curr_b, curr_s = padded.size()
    if curr_s < MAX_LEN:
        pad_tensor = torch.full((curr_b, MAX_LEN - curr_s), PAD_ID, dtype=torch.long)
        padded = torch.cat([padded, pad_tensor], dim=1)
    else:
        padded = padded[:, :MAX_LEN]
    return padded.to(DEVICE)

train_ds = SummarizationDataset("therapml/summarize/dataset/train.csv", tokenizer, MAX_LEN)
valid_ds = SummarizationDataset("therapml/summarize/dataset/valid.csv", tokenizer, MAX_LEN)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)

config = {
    "vocab_size": 64000,
    "context_length": 512,
    "d_model": 768,
    "num_layers": 12,
    "num_heads": 12,
    "d_ff": 3072,
    "num_tokens": 512
}

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
criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID) # Don't train on PADs

best_val_loss = float('inf')

checkpoints = "therapml/summarize/models"

for epoch in range(1, EPOCHS + 1):
    model.train()
    
    indices = torch.randint(0, len(train_ds), (BATCH_SIZE,))
    
    batch_list = [train_ds[i] for i in indices]
    batch = collate_fn(batch_list)
    
    optimizer.zero_grad()
    
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    
    inputs_padded = torch.nn.functional.pad(inputs, (0, 1), value=PAD_ID)
    
    logits = model(inputs_padded) 
    
    logits = logits[:, :-1, :] 
    
    loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    
    loss.backward()
    optimizer.step()

    if epoch % VALIDATION_INTERVAL == 0:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for v_batch in valid_loader:
                v_inputs = v_batch[:, :-1]
                v_targets = v_batch[:, 1:]
                v_inputs_padded = torch.nn.functional.pad(v_inputs, (0, 1), value=PAD_ID)
                
                v_logits = model(v_inputs_padded)[:, :-1, :]
                v_loss = criterion(v_logits.reshape(-1, v_logits.size(-1)), v_targets.reshape(-1))
                val_loss += v_loss.item()
        
        avg_val = val_loss / len(valid_loader)
        print(f"Epoch {epoch} | Val Loss: {avg_val:.4f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), f"{checkpoints}/best_model.pt")
            print(f"New best model saved with val loss: {best_val_loss:.4f}")

    if epoch % SAVE_INTERVAL == 0:
        torch.save(model.state_dict(), f"{checkpoints}/model_epoch_{epoch}.pt")