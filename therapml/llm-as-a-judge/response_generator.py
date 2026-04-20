import torch
import json
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

model_a_base = "gpt2"
model_a_adapter = "therapml/lora/gpt2_eli5_results/final_adapter"

model_b_base = "mistralai/Mistral-7B-v0.1" 
model_b_adapter = "therapml/dpo/dpo_model_adapter"

output_file = "therapml/llm-as-a-judge/evaluation_data.json"

prompts = [
    "Explain the concept of Direct Preference Optimization (DPO) simply.",
    "Why is the sky blue? Explain it to a 5 year old.",
    "What are the main differences between Python and C++?",
    "Give me advice on how to handle burnout at work.",
    "How does machine learning differ from traditional programming?",
    "What is the greenhouse effect and why does it matter?",
    "Explain quantum computing in simple terms.",
    "What are the benefits of regular exercise?",
    "How does photosynthesis work?",
    "What is the difference between AI, machine learning, and deep learning?",
    "Explain the concept of blockchain technology.",
    "What are the main causes of climate change?",
    "How does the internet work at a basic level?",
    "What are best practices for writing clean code?",
    "Explain the importance of cybersecurity in modern business."
]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

def generate_responses(base_id, adapter_path, prompts_list):
    """Loads a model, generates responses for all prompts, and unloads it."""
    print(f"\n--- Loading {base_id} ---")
    
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        base_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    results = []
    print(f"Generating {len(prompts_list)} responses...")
    for text_prompt in prompts_list:
        messages = [{"role": "user", "content": text_prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        results.append(response)

    print("Unloading model and clearing VRAM...")
    del model
    del base_model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return results

print("Starting sequential generation process...")

sft_responses = generate_responses(model_a_base, model_a_adapter, prompts)

dpo_responses = generate_responses(model_b_base, model_b_adapter, prompts)

evaluation_data = []
for i in range(len(prompts)):
    evaluation_data.append({
        "prompt": prompts[i],
        "sft_response": sft_responses[i],
        "dpo_response": dpo_responses[i]
    })

with open(output_file, "w") as f:
    json.dump(evaluation_data, f, indent=4)

print(f"\nGeneration complete! SFT and DPO responses merged and saved to {output_file}.")