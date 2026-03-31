import pandas as pd
import os
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.normalizer = normalizers.Lowercase()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

trainer = trainers.BpeTrainer(
    vocab_size=64000, 
    show_progress=True,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "A", "B", "C", "D"]
)

dataset_path = "/data/rayhansefat/training/training-wheels/therapml/mmlu/dataset/train.csv"
train_df = pd.read_csv(dataset_path)

def get_training_corpus():
    batch_size = 1000
    for i in range(0, len(train_df), batch_size):
        batch = train_df.iloc[i : i + batch_size]
        
        combined = (
            batch['prompt'].astype(str) + " " + 
            batch['A'].astype(str) + " " + 
            batch['B'].astype(str) + " " + 
            batch['C'].astype(str) + " " + 
            batch['D'].astype(str)
        ).fillna("").tolist()
        
        for text in combined:
            yield text

print("Starting tokenizer training...")
tokenizer.train_from_iterator(get_training_corpus(), trainer)

save_path = "therapml/mmlu/tokenizers/my_bpe_tokenizer.json"

os.makedirs(os.path.dirname(save_path), exist_ok=True)

tokenizer.save(save_path)
print(f"Successfully saved tokenizer to: {save_path}")