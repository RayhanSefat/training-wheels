import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset

# Load Model and Tokenizer
model_name = "gpt2"
base_model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token # Set pad token for batching

# Load and Prepare Dataset (using a small subset for demonstration)
dataset = load_dataset("sentence-transformers/eli5", split="train[:1000]") # Use a small slice
validation_dataset = load_dataset("sentence-transformers/eli5", split="train[1000:2000]") # Small validation set

# Basic preprocessing: Tokenize the text
def preprocess_function(examples):
    # Concatenate question and answer for causal LM training
    texts = [q + " " + a for q, a in zip(examples['question'], examples['answer'])]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
tokenized_validation_dataset = validation_dataset.map(preprocess_function, batched=True, remove_columns=validation_dataset.column_names)

print("Base model loaded:", model_name)
print("Dataset loaded and tokenized.")

# Define LoRA configuration
lora_config = LoraConfig(
    r=16,                             # Rank r
    lora_alpha=32,                    # Scaling factor alpha
    target_modules=["c_attn"],        # Apply LoRA to query, key, value projections in attention
    lora_dropout=0.05,                # Dropout probability for LoRA layers
    bias="none",                      # Do not train bias parameters
    task_type=TaskType.CAUSAL_LM      # Task type for causal language modeling
)

print("LoRA Configuration:")
print(lora_config)

# Calculate original trainable parameters
original_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
print(f"Original trainable parameters: {original_params:,}")

# Apply LoRA configuration to the base model
lora_model = get_peft_model(base_model, lora_config)

# Calculate LoRA trainable parameters
lora_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
print(f"LoRA trainable parameters:     {lora_params:,}")

# Calculate the reduction
reduction = (original_params - lora_params) / original_params * 100
print(f"Parameter reduction:           {reduction:.2f}%")

# Print trainable modules for verification
lora_model.print_trainable_parameters()

# Define Training Arguments
output_dir = "therapml/lora/gpt2_eli5_results"
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,             # Keep it short for demonstration
    per_device_train_batch_size=4,  # Adjust based on your GPU memory
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=200,
    learning_rate=2e-4,             # Typical learning rate for LoRA
    fp16=torch.cuda.is_available(), # Use mixed precision if available
    # Add other arguments as needed: weight_decay, warmup_steps, etc.
)

# Data Collator for Causal LM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Initialize Trainer
trainer = Trainer(
    model=lora_model,               # Use the PEFT model
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_validation_dataset,
    # tokenizer=tokenizer,
    data_collator=data_collator,
)

print("Trainer initialized. Starting training...")

# Start training
trainer.train()

# Plot training vs validation loss
import matplotlib.pyplot as plt

# Extract training history
train_loss = trainer.state.log_history

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

print(f"Training loss plot saved to {output_dir}/loss.png")

print("Training finished.")

# Define path to save the adapter
adapter_path = f"{output_dir}/final_adapter"

# Save the LoRA adapter weights
lora_model.save_pretrained(adapter_path)

print(f"LoRA adapter saved to: {adapter_path}")

# You can verify the small size of the saved adapter directory
# !ls -lh {adapter_path}

# Load the base model again (or use the one already in memory)
base_model_reloaded = AutoModelForCausalLM.from_pretrained(model_name)

# Load the PEFT model by merging the adapter weights
inference_model = PeftModel.from_pretrained(base_model_reloaded, adapter_path)

# Ensure the model is in evaluation mode and on the correct device
inference_model.eval()
if torch.cuda.is_available():
    inference_model.to("cuda")

print("Base model loaded and LoRA adapter applied for inference.")

# Example Inference (Optional)
prompt = "What is the main cause of climate change?"
inputs = tokenizer(prompt, return_tensors="pt")

if torch.cuda.is_available():
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

# Generate text
with torch.no_grad():
    outputs = inference_model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n--- Example Generation ---")
print("Prompt:", prompt)
print("Generated:", generated_text)
print("------------------------")
