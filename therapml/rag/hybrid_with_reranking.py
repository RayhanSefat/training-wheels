import json
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

K_INITIAL = 70
K_FINAL = 30
RRF_K = 60

def rrf_score(v_ranks, b_ranks):
    scores = {}
    for r, i in enumerate(v_ranks): scores[i] = scores.get(i, 0) + 1/(RRF_K + r)
    for r, i in enumerate(b_ranks): scores[i] = scores.get(i, 0) + 1/(RRF_K + r)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def hybrid_rerank_retrieve(query, chunks, mapping=None):
    tokenized_corpus = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    b_scores = bm25.get_scores(query.lower().split())
    b_ranks = np.argsort(b_scores)[::-1]

    q_emb = bi_encoder.encode([query], show_progress_bar=False)
    c_embs = bi_encoder.encode(chunks, show_progress_bar=False)
    v_sims = cosine_similarity(q_emb, c_embs).flatten()
    v_ranks = np.argsort(v_sims)[::-1]

    combined_indices = [idx for idx, score in rrf_score(v_ranks, b_ranks)[:K_INITIAL]]
    candidates = [chunks[idx] for idx in combined_indices]

    pairs = [[query, cand] for cand in candidates]
    rerank_scores = reranker.predict(pairs)
    
    reranked_indices = np.argsort(rerank_scores)[::-1]
    top_candidates_indices = [combined_indices[i] for i in reranked_indices[:K_FINAL]]

    if mapping:
        return list(dict.fromkeys([mapping[i] for i in top_candidates_indices]))
    return [chunks[i] for i in top_candidates_indices]

with open("therapml/rag/processed_chunks.json", "r") as f:
    processed_data = json.load(f)

ds = load_dataset("deepmind/narrativeqa", split="train")
target_ids = [d['document_id'] for d in processed_data]
df_qs = ds.filter(lambda x: x['document']['id'] in target_ids).to_pandas()

final_results = []

for doc_entry in processed_data:
    doc_id = doc_entry['document_id']
    doc_qs = df_qs[df_qs['document'].apply(lambda x: x['id']) == doc_id]
    
    res = {"document_id": doc_id, "questions_count": len(doc_qs)}
    
    strategies = doc_entry['strategies']

    for strat_name, content in strategies.items():
        print(f"Evaluating {strat_name} for document {doc_id[:8]}...")

        if strat_name == 'parent_document':
            search_texts, mapping = [], {}
            for item in content:
                for child in item['children']:
                    search_texts.append(child); mapping[len(search_texts)-1] = item['parent']
        else:
            search_texts, mapping = content, None

        hits = 0
        for _, q_row in doc_qs.iterrows():
            retrieved = hybrid_rerank_retrieve(q_row['question']['text'], search_texts, mapping)
            
            for ans in q_row['answers']:
                if any(ans['text'].lower() in c.lower() for c in retrieved):
                    hits += 1; break
        
        res[strat_name] = hits / len(doc_qs)
    final_results.append(res)

final_results_df = pd.DataFrame(final_results)
print("\n--- Final Performance with Re-ranking ---")
print(final_results_df)

final_results_df.to_csv("therapml/rag/performance/hybrid_with_reranking_performance.csv", index=False)