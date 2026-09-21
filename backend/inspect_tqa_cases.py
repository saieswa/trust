import pandas as pd
from evaluation.metrics import evaluate_truthfulqa_case, token_f1

df = pd.read_csv('evaluation/reports/raw_results.csv')
tqa = df[df['dataset'] == 'TruthfulQA']
for idx, r in tqa.iterrows():
    print(f"=== {r['id']}: {r['question']} ===")
    print(f"Ground Truth: {r['ground_truth']}")
    print(f"Normal Ans: {r['normal_answer']} | Normal Correct: {r['normal_correct']}")
    print(f"Trust Ans: {r['trust_answer']}")
    print(f"Trust Correct: {r['trust_correct']} | Trust Hallu: {r['trust_hallucinated']} | Trust Abs: {r['trust_abstained']}")
    print("-" * 50)
