from huggingface_hub import notebook_login

notebook_login()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "mistralai/Mistral-7B-Instruct-v0.1"

# Configure quantization to load the model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load the model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True
)
model.config.use_cache = False # Disable cache for training

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # Set padding token

from datasets import load_dataset

# Load a sample of the dataset
dataset = load_dataset("databricks/dolly-v2-12k", split="train[:1000]") # dataset not found

# Function to format the prompts
def format_prompt(sample):
    instruction = sample["instruction"]
    context = sample["context"]
    response = sample["response"]
    if context:
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{context}

### Response:
{response}"""
    else:
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}"""
    return {"text": prompt}

# Apply the formatting
formatted_dataset = dataset.map(format_prompt)