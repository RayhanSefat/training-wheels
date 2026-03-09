"""
This file is to be run to create the my_tokenizer.json file
"""

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from datasets import load_dataset

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

tokenizer.normalizer = normalizers.Lowercase()

tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.WordPieceTrainer(
    vocab_size=30000, 
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

train_dataset = load_dataset("roneneldan/TinyStories")["train"]
tokenizer.train_from_iterator(train_dataset["text"], trainer)

tokenizer.save("therapml/train_model/tokenizers/my_tokenizer.json")