from datasets import load_dataset
dataset = load_dataset("roneneldan/TinyStories")

train_dataset = dataset["train"]
valid_dataset = dataset["validation"]

CHECKPOINT_FOLDER = "therapml/train_model/models"

block_size = 512
batch_size = 32
d_model = 128
num_layers = 16
num_heads = 4
d_ff = 512