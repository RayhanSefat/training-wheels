from huggingface_hub import notebook_login

notebook_login()

from datasets import load_dataset

# Load a small part of the training set
train_dataset = load_dataset("squad", split="train[:5000]")

# Split the 5000 examples into training and validation sets
train_test_split = train_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Define the formatting function
def format_qa(example):
    # SQuAD dataset has answers as a list, take the first one
    question = example["question"]
    answer = example["answers"]["text"][0]
    return f"question: {question} answer: {answer}"

from transformers import AutoTokenizer

# Load the tokenizer for Qwen/Qwen2.5-0.5B
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token # Set padding token

def tokenize_function(examples):
    formatted_examples = [
        format_qa({"question": q, "answers": a})
        for q, a in zip(examples["question"], examples["answers"])
    ]
    tokenized = tokenizer(formatted_examples, truncation=True, padding="max_length", max_length=128)
    tokenized["labels"] = tokenized["input_ids"].copy()  # Add labels for training
    return tokenized

# Apply tokenization to the datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)

from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

# Load the pre-trained model
model = AutoModelForCausalLM.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir="therapml/fine-tuning/qwen2.5-0.5b-squad-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    load_best_model_at_end=True,
    push_to_hub=False, # Set to True if you are logged in and want to push
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    processing_class=tokenizer,
)

# Start fine-tuning
trainer.train()

import math
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

# Save the model and tokenizer
trainer.save_model("therapml/fine-tuning/my_finetuned_qwen2.5-0.5b")
tokenizer.save_pretrained("therapml/fine-tuning/my_finetuned_qwen2.5-0.5b")