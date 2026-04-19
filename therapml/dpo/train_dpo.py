from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

model_name = "gpt2"
base_model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token # Set pad token for batching

# Load and Prepare Dataset (using a small subset for demonstration)
dataset = load_dataset("Anthropic/hh-rlhf", split="train[:1000]") # Use a small slice
validation_dataset = load_dataset("Anthropic/hh-rlhf", split="train[1000:2000]") # Small validation set

# Basic preprocessing: Tokenize the text
def preprocess_function(examples):
    batch_size = len(examples["chosen"])
    return {
        "prompt": ["This is a dummy prompt."] * batch_size,  # Placeholder prompt for DPO training
        "chosen": examples["chosen"],
        "rejected": examples["rejected"]
    }

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
tokenized_validation_dataset = validation_dataset.map(preprocess_function, batched=True, remove_columns=validation_dataset.column_names)

output_dir = "therapml/dpo/gpt2_hh-rlhf_results"
training_args = DPOConfig(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    remove_unused_columns=False,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=20,
    save_strategy="epoch",
    max_length=512
    # bf16=True,  # Use mixed precision if supported
)

dpo_trainer = DPOTrainer(
    model=base_model,
    ref_model=None,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_validation_dataset,
    processing_class=tokenizer
)

dpo_trainer.train()

# Plot training vs validation loss
import matplotlib.pyplot as plt

# Extract training history
train_loss = dpo_trainer.state.log_history

steps = [log['step'] for log in train_loss if 'loss' in log]
losses = [log['loss'] for log in train_loss if 'loss' in log]
val_steps = [log['step'] for log in train_loss if 'eval_loss' in log]
val_losses = [log['eval_loss'] for log in train_loss if 'eval_loss' in log]

plt.figure(figsize=(10, 6))
plt.plot(steps, losses, label='Training Loss', marker='o')
plt.plot(val_steps, val_losses, label='Validation Loss', marker='s')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss Over Steps')
plt.legend()
plt.grid(True)
plt.savefig(f'{output_dir}/loss.png')
plt.close()