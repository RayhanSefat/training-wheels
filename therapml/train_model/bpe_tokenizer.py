"""
This file is to be run to create the my_bpe_tokenizer.json file
"""

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from datasets import load_dataset

tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

tokenizer.normalizer = normalizers.Lowercase()

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

trainer = trainers.BpeTrainer(
    vocab_size=40000, 
    show_progress=True,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

train_dataset = load_dataset("roneneldan/TinyStories")["train"]

def get_training_corpus():
    for i in range(0, len(train_dataset), 1000):
        yield train_dataset[i : i + 1000]["text"]

tokenizer.train_from_iterator(get_training_corpus(), trainer)

tokenizer.save("therapml/train_model/tokenizers/my_bpe_tokenizer.json")