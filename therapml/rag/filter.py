import pandas as pd
import json
from datasets import Dataset, load_dataset

ds = load_dataset("deepmind/narrativeqa", split="train")

df = ds.to_pandas()

df['doc_text'] = df['document'].apply(lambda x: x['text'])

top_10_doc_texts = df['doc_text'].value_counts().nlargest(10).index

filtered_df = df[df['doc_text'].isin(top_10_doc_texts)]

top_10_ds = Dataset.from_pandas(filtered_df)

print(f"Original rows: {len(df)}")
print(f"Filtered rows: {len(filtered_df)}")
print(f"Unique documents remaining: {len(filtered_df['doc_text'].unique())}")

unique_docs_df = filtered_df.drop_duplicates(subset=['doc_text'])

unique_docs_list = unique_docs_df['document'].to_list()

def handle_serialization(obj):
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

with open("therapml/rag/top_10_documents.json", "w", encoding="utf-8") as f:
    json.dump(unique_docs_list, f, indent=4, default=handle_serialization)

print("Saved successfully!")