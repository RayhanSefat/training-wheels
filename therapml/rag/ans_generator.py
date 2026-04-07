import json
from groq import Groq
from therapml.summarize.secrete import GROQ_API_KEY
import random

with open("therapml/rag/retrieved_chunks_cache.json", "r") as f:
    data = json.load(f)

client = Groq(
    api_key=GROQ_API_KEY
)

evaluation_results = []

q_info_list = []

for doc in data:
    for q_id, q_info in doc["questions"].items():
        q_info_list.append(q_info)

random.shuffle(q_info_list)

for i in range(1):
    q_info = q_info_list[i]
    question_text = q_info["question"]
    ground_truth = q_info["ground_truth"]

    print(f"\nProcessing question: {question_text}")
    print(f"Ground truth answer: {ground_truth}")
        
    semantic_chunks = q_info["strategy"].get("semantic", [])
    context_str = "\n\n".join(semantic_chunks)

    prompt = f"""Answer the question based only on the provided context (Remember, the answer must be precise and to the point, without any hallucination.):
        Context: {context_str}
        Question: {question_text}
        Answer:"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        generated_answer = response.choices[0].message.content.strip()
    except Exception as e:
        generated_answer = f"Error generating answer: {e}"

    print(f"Generated answer: {generated_answer}")
        
    evaluation_results.append({
        "question": question_text,
        "contexts": context_str,
        "answer": generated_answer,
        "ground_truth": ground_truth
    })

with open("therapml/rag/evaluation_results_sample.json", "w") as f:
    json.dump(evaluation_results, f, indent=2)