"""
This file is to be run to create the my_tokenizer.json file
"""

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

tokenizer.normalizer = normalizers.Lowercase()

tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.WordPieceTrainer(
    vocab_size=30000, 
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

files = [
    "therapml/train_model/dataset/train.csv",
    "therapml/train_model/dataset/validation.csv"
]
tokenizer.train(files, trainer)

tokenizer.save("therapml/train_model/tokenizers/my_tokenizer.json")