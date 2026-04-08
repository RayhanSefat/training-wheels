from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Load the fine-tuned model for inference
finetuned_model = AutoModelForCausalLM.from_pretrained("therapml/fine-tuning/my_finetuned_qwen2.5-0.5b")
finetuned_tokenizer = AutoTokenizer.from_pretrained("therapml/fine-tuning/my_finetuned_qwen2.5-0.5b")

# Create a text generation pipeline
generator = pipeline("text-generation", model=finetuned_model, tokenizer=finetuned_tokenizer)

# A new question in the format our model expects
prompt = "question: What is the main purpose of the immune system?"

# Generate an answer
result = generator(prompt, max_length=100, num_return_sequences=1)
print(result[0]['generated_text'])