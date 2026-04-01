import torch
import pandas as pd
import evaluate
from therapml.lm import TransformerLM
from tokenizers import Tokenizer, decoders
from groq import Groq
from therapml.train_model.generator import generate_dummy_weights
from .secrete import GROQ_API_KEY

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "therapml/summarize/models/best_model.pt"
TOKENIZER_PATH = "therapml/summarize/tokenizers/my_bpe_tokenizer.json"
TEST_CSV = "therapml/summarize/dataset/test.csv"
MAX_LEN = 512

# 1. Load Metrics
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

# 2. Load Tokenizer & Model
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
tokenizer.decoder = decoders.ByteLevel()
config = {
    "vocab_size": 64000,
    "context_length": 512,
    "d_model": 768,
    "num_layers": 12,
    "num_heads": 12,
    "d_ff": 3072,
    "num_tokens": 512
}

weights = generate_dummy_weights(config["vocab_size"], d_model=config["d_model"], d_ff=config["d_ff"], num_layers=config["num_layers"])

model = TransformerLM(
    vocab_size=config["vocab_size"],
    context_length=config["context_length"],
    d_model=config["d_model"],
    num_layers=config["num_layers"],
    num_heads=config["num_heads"],
    d_ff=config["d_ff"],
    num_tokens=config["num_tokens"],
    weights=weights
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# 3. Generation Function (Greedy Decoding)
def generate_summary(dialogue, max_new_tokens=30, temperature=0.7):
    # 1. Prepare the prompt
    prompt = f"Dialogue: {dialogue} Summary:"
    input_ids = tokenizer.encode(prompt).ids
    
    # Track the full sequence of IDs
    generated_ids = list(input_ids)
    
    # Stop tokens
    sep_id = tokenizer.token_to_id("[SEP]")
    pad_id = tokenizer.token_to_id("[PAD]")

    for _ in range(max_new_tokens):
        # 2. Prepare the window (exactly 512 tokens)
        # We take the most recent context that fits
        curr_context = generated_ids[-MAX_LEN:]
        curr_len = len(curr_context)
        
        # Manually pad to 512 to satisfy your model's internal mask
        if curr_len < MAX_LEN:
            padded_input = curr_context + [pad_id] * (MAX_LEN - curr_len)
        else:
            padded_input = curr_context
            
        input_tensor = torch.tensor([padded_input]).to(DEVICE)
        
        # 3. Forward Pass
        with torch.no_grad():
            logits = model(input_tensor) # Shape: (1, 512, vocab_size)
            
            # CRITICAL: We need the logit for the LAST ACTUAL TOKEN in the context,
            # not the last token of the 512-padding.
            last_token_idx = curr_len - 1
            next_token_logits = logits[0, last_token_idx, :] / temperature
            
            # 4. Sample the next token
            # Using argmax (greedy) or multinomial (sampling)
            # Let's use argmax for stability first
            next_token = torch.argmax(next_token_logits).item()
            
        # 5. Append and Check for stop sequence
        generated_ids.append(next_token)
            
    # 6. Extraction: Only take the IDs that came AFTER the prompt
    summary_ids = generated_ids[len(input_ids):]
    
    # Filter out any accidental PADs or SEPs from the string
    decoded_output = tokenizer.decode(summary_ids).strip()
    
    return decoded_output

# 4. LLM-as-a-Judge Function
client = Groq(api_key=GROQ_API_KEY)

def get_llm_grade(dialogue, our_summary, gold_summary):
    prompt = f"""
    Evaluate two summaries of the following dialogue:
    Dialogue: {dialogue}
    
    Summary A (Model-1): {our_summary}
    Summary B (Model-2): {gold_summary}
    
    Grade Summary A and Summary B on a scale of 1-5 for:
    1. Coherence
    2. Informativeness
    3. Hallucination (1 is high hallucination, 5 is no hallucination)
    
    Finally, state which summary is better and why.
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile", # High intelligence for a judge
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2 # Lower temperature for objective grading
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Grading Error: {e}"

# 5. Main Execution
test_df = pd.read_csv(TEST_CSV)
samples = test_df.sample(5)

with open("therapml/summarize/summary_evaluation_results.txt", "w") as f:
    for i, (_, row) in enumerate(samples.iterrows()):
        dialogue = str(row['dialogue']).strip('"')
        gold_summary = str(row['summary']).strip('"')
        
        # Generate
        our_summary = generate_summary(dialogue)
        
        # Calculate Scores
        r_score = rouge.compute(predictions=[our_summary], references=[gold_summary])['rougeL']
        b_score = 0.0
        try:
            b_score = bleu.compute(predictions=[our_summary.split()], references=[[gold_summary.split()]])['bleu']
        except:
            b_score = 0.0
        
        # Get LLM Grade
        grade_report = get_llm_grade(dialogue, our_summary, gold_summary)
        
        # Save Results
        f.write(f"--- SAMPLE {i+1} ---\n")
        f.write(f"DIALOGUE:\n{dialogue}\n\n")
        f.write(f"OUR SUMMARY:\n{our_summary}\n\n")
        f.write(f"GOLD SUMMARY:\n{gold_summary}\n\n")
        f.write(f"ROUGE-L: {r_score:.4f} | BLEU: {b_score:.4f}\n\n")
        f.write(f"LLM EVALUATION:\n{grade_report}\n")
        f.write("-" * 50 + "\n\n\n\n")

print("Evaluation complete. Results saved to summary_evaluation_results.txt")