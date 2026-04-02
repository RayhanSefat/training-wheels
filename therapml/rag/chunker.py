import json
import pandas as pd

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

with open("therapml/rag/top_10_documents.json", "r", encoding="utf-8") as f:
    docs_json = json.load(f)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

chunk_size = 1000
chunk_overlap = 100

fixed_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", " ", ""]
)

semantic_splitter = SemanticChunker(embeddings)


all_processed_data = []

child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

def process_entry(doc_entry):
    text = doc_entry['text']
    doc_id = doc_entry.get('id', 'unknown')
    
    fixed_chunks = fixed_splitter.split_text(text)
    recursive_chunks = recursive_splitter.split_text(text)
    semantic_chunks = semantic_splitter.split_text(text)
    
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    parent_chunks = recursive_splitter.split_text(text)
    total_child_chunks = 0
    for p in parent_chunks:
        total_child_chunks += len(child_splitter.split_text(p))

    parent_child_pairs = []
    for p_text in recursive_chunks:
        children = child_splitter.split_text(p_text)
        parent_child_pairs.append({
            "parent": p_text,
            "children": children
        })

    doc_data = {
        "document_id": doc_id,
        "metadata": {
            "title": entry.get('summary', {}).get('title', 'Unknown'),
            "original_word_count": entry.get('word_count')
        },
        "strategies": {
            "fixed": fixed_chunks,
            "recursive": recursive_chunks,
            "semantic": semantic_chunks,
            "parent_document": parent_child_pairs
        }
    }
    all_processed_data.append(doc_data)

    return {
        "ID": doc_id,
        "Fixed_Count": len(fixed_chunks),
        "Recursive_Count": len(recursive_chunks),
        "Semantic_Count": len(semantic_chunks),
        "Parent_Doc_Pairs": f"{len(parent_chunks)} Parents / {total_child_chunks} Children"
    }

results = []

for entry in docs_json:
    print(f"Processing {entry.get('id')[:8]}...")
    results.append(process_entry(entry))


df_results = pd.DataFrame(results)
print("\n\n\n--------- Chunking Strategy Comparison ---------")
print(df_results.to_string(index=False))

df_results.to_csv("therapml/rag/chunking_comparison.csv", index=False)
with open("therapml/rag/processed_chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_processed_data, f, indent=4)