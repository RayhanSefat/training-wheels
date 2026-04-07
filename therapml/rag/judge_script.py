import os
import json
import pandas as pd
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import GeminiModel
from .secrete import GEMINI_API_KEY

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

gemini_judge = GeminiModel(
    model="gemini-2.5-flash"
)

# Load the data
with open("therapml/rag/evaluation_results_sample.json", "r") as f:
    evaluation_results = json.load(f)

eval_df = pd.DataFrame(evaluation_results)
eval_df["contexts"] = eval_df["contexts"].apply(lambda x: [x] if isinstance(x, str) else x)

test_cases = []
for _, row in eval_df.iterrows():
    test_case = LLMTestCase(
        input=row["question"],
        actual_output=row["answer"],
        retrieval_context=row["contexts"],
        expected_output=row["ground_truth"]
    )
    test_cases.append(test_case)

faithfulness = FaithfulnessMetric(threshold=0.5, model=gemini_judge)
context_precision = ContextualPrecisionMetric(threshold=0.5, model=gemini_judge)

results = evaluate(
    test_cases=test_cases,
    metrics=[faithfulness, context_precision]
)

summary_data = []

for result in results.test_results:
    scores = {metric.name: metric.score for metric in result.metrics_data}
    summary_data.append(scores)

df_final = pd.DataFrame(summary_data)

df_final = df_final.rename(columns={
    "Faithfulness": "faithfulness", 
    "Contextual Precision": "context_precision"
})

print("\n--- RAG Performance per Chunking Strategy ---")
print(df_final)