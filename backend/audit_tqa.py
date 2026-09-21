import csv
import json

csv_path = r"c:\Users\ADMIN\Downloads\trust-aware-rag\trust-aware-rag\backend\evaluation\reports\raw_results.csv"

with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

tqa_rows = [r for r in rows if r["dataset"] == "TruthfulQA"]
print(f"Total TruthfulQA rows: {len(tqa_rows)}")

print("\n--- Summary of TruthfulQA Cases ---")
for i, r in enumerate(tqa_rows):
    print(f"\n[{i+1}] ID: {r['id']}")
    print(f"Question: {r['question']}")
    print(f"Ground Truth: {r['ground_truth']}")
    print(f"Normal Answer: {r['normal_answer']}")
    print(f"Normal Correct: {r['normal_correct']} | Normal Hallu: {r['normal_hallucinated']} | Normal F1: {r['normal_f1']}")
    print(f"Trust Answer: {r['trust_answer'][:150]}..." if len(r['trust_answer']) > 150 else f"Trust Answer: {r['trust_answer']}")
    print(f"Trust Correct: {r['trust_correct']} | Trust Hallu: {r['trust_hallucinated']} | Trust Abstained: {r['trust_abstained']}")
    print(f"Trust Score: {r['trust_score']} | Verif Status: {r['verification_status']} | Contra Count: {r['contradiction_count']} | Retrieval Rel: {r['retrieval_relevance']}")
