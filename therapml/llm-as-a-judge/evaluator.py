import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

input_file = "therapml/llm-as-a-judge/evaluation_data.json"
judge_model_id = "Qwen/Qwen2.5-7B-Instruct"

with open(input_file, "r") as f:
    evaluation_data = json.load(f)

print(f"Loading local judge model ({judge_model_id}) in 4-bit...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(judge_model_id)
model = AutoModelForCausalLM.from_pretrained(
    judge_model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
model.eval()

JUDGE_SYSTEM_PROMPT = """You are an impartial judge evaluating the quality of AI assistant responses. 
Your primary metric is 'helpfulness'. A helpful response directly addresses the user's prompt, is accurate, clear, and appropriately detailed.
"""

JUDGE_USER_PROMPT = """
[User Prompt]
{prompt}

[Model A Response]
{response_a}

[Model B Response]
{response_b}

[Evaluation Instructions]
Evaluate both models based on helpfulness. 
1. Provide a brief explanation of your reasoning.
2. Conclude your evaluation by strictly outputting one of the following exact phrases on a new line at the very end:
"Winner: Model A"
"Winner: Model B"
"Tie"
"""

results = {"Model A": 0, "Model B": 0, "Tie": 0}

print(f"\nStarting Local Evaluation on {len(evaluation_data)} prompts...\n")

for i, data in enumerate(evaluation_data):
    print(f"Evaluating Prompt {i+1}/{len(evaluation_data)}...")
    
    formatted_prompt = JUDGE_USER_PROMPT.format(
        prompt=data["prompt"],
        response_a=data["sft_response"],
        response_b=data["dpo_response"]
    )

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": formatted_prompt}
    ]
    
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.01,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    judge_output = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    
    if "Winner: Model A" in judge_output:
        results["Model A"] += 1
        winner = "Model A (SFT)"
    elif "Winner: Model B" in judge_output:
        results["Model B"] += 1
        winner = "Model B (DPO)"
    else:
        results["Tie"] += 1
        winner = "Tie"

    print(f"Decision: {winner}")
    print(f"Judge Reasoning:\n{judge_output}\n")
    print("-" * 50)

print("\n=== FINAL RESULTS ===")
print(f"Total Prompts Evaluated: {len(evaluation_data)}")
print(f"Model A (SFT) Wins: {results['Model A']}")
print(f"Model B (DPO) Wins: {results['Model B']}")
print(f"Ties: {results['Tie']}")

if results['Model B'] > results['Model A']:
    print("\nConclusion: DPO successfully improved helpfulness!")
elif results['Model A'] > results['Model B']:
    print("\nConclusion: The base SFT model was preferred. You may need to tune the DPO beta parameter.")
else:
    print("\nConclusion: Both models performed equally well on this dataset.")