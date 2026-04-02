import json
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

with open("therapml/rag/top_10_documents.json", "r", encoding="utf-8") as f:
    top_docs = json.load(f)

with open("therapml/rag/processed_chunks.json", "r", encoding="utf-8") as f:
    processed_chunks = json.load(f)

top_doc_ids = [doc['id'] for doc in top_docs]

print("Loading NarrativeQA dataset...")
ds = load_dataset("deepmind/narrativeqa", split="train")

filtered_qs = ds.filter(lambda x: x['document']['id'] in top_doc_ids)
df_questions = filtered_qs.to_pandas()

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_recall_score(retrieved_chunks, answers):
    """
    Checks if the ground-truth answer string is present 
    in any of the top-k retrieved chunks.
    """
    for ans in answers:
        ans_text = ans['text'].lower()
        for chunk in retrieved_chunks:
            if ans_text in chunk.lower():
                return 1
    return 0

k = 30
results = []

for doc_entry in processed_chunks:
    doc_id = doc_entry['document_id']
    doc_qs = df_questions[df_questions['document'].apply(lambda x: x['id']) == doc_id]
    
    if doc_qs.empty:
        continue
    
    doc_results = {"document_id": doc_id, "questions_count": len(doc_qs)}
    strategies = doc_entry['strategies']

    for strat_name, content in strategies.items():
        print(f"Evaluating {strat_name} for document {doc_id[:8]}...")
        
        if strat_name == "parent_document":
            children = []
            child_to_parent = {}
            for pair in content:
                for child in pair['children']:
                    children.append(child)
                    child_to_parent[len(children)-1] = pair['parent']
            
            chunk_texts = children
            mapping = child_to_parent
        else:
            chunk_texts = content
            mapping = None

        if not chunk_texts:
            doc_results[strat_name] = 0.0
            continue

        chunk_embeddings = model.encode(chunk_texts)
        
        hits = 0
        for _, q_row in doc_qs.iterrows():
            q_emb = model.encode([q_row['question']['text']])
            sims = cosine_similarity(q_emb, chunk_embeddings).flatten()
            
            top_indices = np.argsort(sims)[-k:][::-1]
            
            if mapping:
                retrieved = list(set([mapping[idx] for idx in top_indices]))
            else:
                retrieved = [chunk_texts[idx] for idx in top_indices]
            
            hits += get_recall_score(retrieved, q_row['answers'])
        
        doc_results[strat_name] = hits / len(doc_qs)

    results.append(doc_results)

report_df = pd.DataFrame(results)
print(f"\n--- Average Retrieval Recall (Recall@{k}) ---")
print(report_df.to_string(index=False))

report_df.to_csv("therapml/rag/performance/retriever_1_performance.csv", index=False)