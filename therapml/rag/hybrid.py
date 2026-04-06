import json
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

K = 10
RRF_K = 60
MODEL_NAME = 'all-MiniLM-L6-v2'
model = SentenceTransformer(MODEL_NAME)

def calculate_rrf(vector_ranks, bm25_ranks):
    """Combines ranks using Reciprocal Rank Fusion."""
    rrf_scores = {}
    for rank, idx in enumerate(vector_ranks):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + (1.0 / (RRF_K + rank))
    for rank, idx in enumerate(bm25_ranks):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + (1.0 / (RRF_K + rank))
    
    sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, score in sorted_indices[:K]]

def check_recall(retrieved_chunks, answers):
    """Returns 1 if any answer text is found in any retrieved chunk."""
    for ans in answers:
        ans_text = ans['text'].lower()
        for chunk in retrieved_chunks:
            if ans_text in chunk.lower():
                return 1
    return 0

with open("therapml/rag/processed_chunks.json", "r") as f:
    processed_data = json.load(f)

ds = load_dataset("deepmind/narrativeqa", split="train")
target_ids = [d['document_id'] for d in processed_data]
df_qs = ds.filter(lambda x: x['document']['id'] in target_ids).to_pandas()

all_results = []

for doc_entry in processed_data:
    doc_id = doc_entry['document_id']
    doc_qs = df_qs[df_qs['document'].apply(lambda x: x['id']) == doc_id]
    
    if doc_qs.empty: continue
    
    stats = {"document_id": doc_id, "questions_count": len(doc_qs)}
    
    strategies = doc_entry['strategies']

    for strat_name, content in strategies.items():
        print(f"Evaluating {strat_name} for document {doc_id[:8]}...")

        if strat_name == 'parent_document':
            search_texts = []
            mapping = {}
            for item in content:
                for child in item['children']:
                    search_texts.append(child)
                    mapping[len(search_texts)-1] = item['parent']
        else:
            search_texts = content
            mapping = None

        if not search_texts:
            stats[strat_name] = 0.0
            continue

        tokenized_corpus = [str(d).lower().split() for d in search_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        chunk_embeddings = model.encode(search_texts, show_progress_bar=False)

        total_hits = 0
        for _, row in doc_qs.iterrows():
            query = row['question']['text']
            ans_list = row['answers']
            
            q_emb = model.encode([query], show_progress_bar=False)
            v_sims = cosine_similarity(q_emb, chunk_embeddings).flatten()
            v_ranks = np.argsort(v_sims)[::-1]
            
            bm_scores = bm25.get_scores(query.lower().split())
            bm_ranks = np.argsort(bm_scores)[::-1]
            
            top_indices = calculate_rrf(v_ranks, bm_ranks)
            
            if mapping:
                retrieved = list(dict.fromkeys([mapping[idx] for idx in top_indices]))
            else:
                retrieved = [search_texts[idx] for idx in top_indices]
            
            total_hits += check_recall(retrieved, ans_list)
        
        stats[strat_name] = total_hits / len(doc_qs)
    
    all_results.append(stats)

results_df = pd.DataFrame(all_results)
print(f"\n--- Final Recall@{K} Comparison (Hybrid Search: Vector + BM25 + RRF) ---")
print(results_df.to_string(index=False))

results_df.to_csv("therapml/rag/performance/hybrid_retrieval_performance.csv", index=False)