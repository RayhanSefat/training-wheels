from datasets import load_dataset

# Load the dataset from a local file or the Hugging Face Hub
# For this example, we'll assume it's available on the Hub.
raw_dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

# Let's see the structure and a few examples
print(raw_dataset)
print(raw_dataset[0])

# Filter out examples where the instruction is too short
initial_size = len(raw_dataset)
filtered_dataset = raw_dataset.filter(lambda example: len(example['instruction']) > 3)

# Filter out examples where the response is too short
filtered_dataset = filtered_dataset.filter(lambda example: len(example['response']) > 3)

print(f"Original size: {initial_size}")
print(f"Size after filtering: {len(filtered_dataset)}")

def format_prompt(example):
    """Formats a single example into a standardized prompt string."""
    instruction = f"### Instruction:\n{example['instruction']}"

    # Use context if it exists and is not empty
    if example.get('context') and example['context'].strip():
        context = f"### Input:\n{example['context']}"
    else:
        context = ""

    response = f"### Response:\n{example['response']}"

    # Join the parts, filtering out empty strings
    full_prompt = "\n\n".join(filter(None, [instruction, context, response]))

    return {"text": full_prompt}

# Apply the formatting function to each example
structured_dataset = filtered_dataset.map(format_prompt)

# Let's inspect an example with context and one without
print("--- Example with context ---")
print(structured_dataset[0]['text'])

print("\n--- Example without context ---")
# Find an example without context to print
for ex in structured_dataset:
    if "### Input:" not in ex['text']:
        print(ex['text'])
        break

from transformers import AutoTokenizer

# Load the tokenizer for a specific model
model_id = "mistralai/Mistral-7B-v0.1"
# Note: You may need to authenticate with Hugging Face to access this model
# from huggingface_hub import login; login()
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Set a padding token if one is not already defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    # Append the EOS token to the end of each text
    text_with_eos = [s + tokenizer.eos_token for s in examples["text"]]

    # Tokenize the text
    return tokenizer(
        text_with_eos,
        truncation=True,        # Truncate sequences longer than max length
        max_length=512,         # A common max length
        padding=False,          # We will handle padding later with a data collator
    )

# Apply the tokenization
tokenized_dataset = structured_dataset.map(
    tokenize_function, 
    batched=True,  # Process examples in batches for efficiency
    remove_columns=structured_dataset.column_names # Remove old text columns
)

# Check the output of the tokenization
print(tokenized_dataset[0].keys())
print(tokenized_dataset[0]['input_ids'][:20]) # Print first 20 token IDs

# Save the dataset to a local directory
tokenized_dataset.save_to_disk("therapml/fine-tuning/fine-tuning-dataset")

# You can load it back anytime using:
# from datasets import load_from_disk
# reloaded_dataset = load_from_disk("./fine-tuning-dataset")
