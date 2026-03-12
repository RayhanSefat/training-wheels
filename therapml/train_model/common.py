from datasets import load_dataset
dataset = load_dataset("Skylion007/openwebtext")["train"]

train_test = dataset.train_test_split(test_size=0.1, seed=42)
train_val = train_test["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = train_val["train"]
valid_dataset = train_val["test"]
test_dataset = train_test["test"]

CHECKPOINT_FOLDER = "therapml/train_model/models_7"

block_size = 256
batch_size = 96
d_model = 128
num_layers = 16
num_heads = 4
d_ff = 512