import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

model_id = "mistralai/Mistral-7B-v0.1" 
dataset_id = "llmat/dpo-orpo-mix-38k-balanced"
output_dir = "therapml/dpo/dpo_model_adapter"

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

if tokenizer.chat_template is None:
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
    )

print("Loading and formatting dataset...")
dataset = load_dataset(dataset_id, split="train")

def prep_dpo_data(example):
    """
    DPOTrainer requires exactly three columns: 'prompt', 'chosen', and 'rejected'.
    This function handles standard conversational preference structures.
    """
    if isinstance(example.get('chosen'), list):
        prompt_msgs = [m for m in example['chosen'] if m['role'] != 'assistant']
        prompt = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        
        chosen = [m['content'] for m in example['chosen'] if m['role'] == 'assistant'][-1]
        rejected = [m['content'] for m in example['rejected'] if m['role'] == 'assistant'][-1]
    else:
        prompt = example['prompt']
        chosen = example['chosen']
        rejected = example['rejected']
        
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

dataset = dataset.map(prep_dpo_data, remove_columns=dataset.column_names)

dataset = dataset.train_test_split(test_size=0.05, seed=42)

print("Loading model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] 
)

training_args = DPOConfig(
    output_dir=output_dir,
    beta=0.1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    max_steps=500,
    save_strategy="steps",
    save_steps=100,
    logging_steps=10,
    optim="paged_adamw_32bit",
    fp16=False,
    bf16=True,
    remove_unused_columns=False,
    report_to="none"
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    processing_class=tokenizer,
    peft_config=peft_config,
)

print("Starting DPO training...")
trainer.train()

trainer.save_model(output_dir)
print(f"Training complete. Adapter saved to {output_dir}")