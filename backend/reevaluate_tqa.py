"""
Re-evaluate the existing raw_results.csv with the fixed evaluate_truthfulqa_case function.
This shows what results would look like if only the evaluator was fixed (no new LLM calls).
"""
import pandas as pd
import json
from evaluation.metrics import evaluate_truthfulqa_case

df = pd.read_csv('evaluation/reports/raw_results.csv')

# Load TruthfulQA metadata (we need correct_answers and incorrect_answers)
from evaluation.datasets.truthfulqa_loader import load_truthfulqa
items = load_truthfulqa(limit=25, seed=42)
item_map = {item.id: item for item in items}

tqa_rows = df[df['dataset'] == 'TruthfulQA'].copy()
print(f"Re-evaluating {len(tqa_rows)} TruthfulQA cases with fixed evaluator...\n")

corrected = []
for idx, row in tqa_rows.iterrows():
    item = item_map.get(row['id'])
    if not item:
        continue
    
    tqa_meta = dict(item.metadata)
    tqa_meta['ground_truth'] = item.ground_truth
    
    normal_ans = str(row['normal_answer']) if pd.notna(row['normal_answer']) else ''
    trust_ans = str(row['trust_answer']) if pd.notna(row['trust_answer']) else ''
    
    tqa_normal = evaluate_truthfulqa_case(normal_ans, tqa_meta)
    
    if row['trust_abstained']:
        trust_correct = False
        trust_hallu = False
    else:
        tqa_trust = evaluate_truthfulqa_case(trust_ans, tqa_meta)
        trust_correct = tqa_trust['is_truthful']
        trust_hallu = tqa_trust['is_hallucinated']
    
    old_nc = bool(row['normal_correct'])
    old_tc = bool(row['trust_correct'])
    new_nc = tqa_normal['is_truthful']
    new_tc = trust_correct
    
    changed = (old_nc != new_nc) or (old_tc != new_tc)
    corrected.append({
        'id': row['id'],
        'question': row['question'],
        'ground_truth': item.ground_truth,
        'normal_answer': normal_ans[:80],
        'trust_answer': trust_ans[:80],
        'old_normal_correct': old_nc,
        'new_normal_correct': new_nc,
        'old_trust_correct': old_tc,
        'new_trust_correct': new_tc,
        'trust_abstained': bool(row['trust_abstained']),
        'changed': changed,
    })
    if changed:
        print(f"[CHANGED] {row['id']}")
        print(f"  Q: {row['question']}")
        print(f"  GT: {item.ground_truth}")
        print(f"  Normal: {old_nc} -> {new_nc} | Trust: {old_tc} -> {new_tc}")
        print()

total = len(corrected)
old_n_acc = sum(r['old_normal_correct'] for r in corrected) / total * 100
old_t_acc = sum(r['old_trust_correct'] for r in corrected) / total * 100
new_n_acc = sum(r['new_normal_correct'] for r in corrected) / total * 100
new_t_acc = sum(r['new_trust_correct'] for r in corrected) / total * 100

print("=" * 60)
print("TruthfulQA Re-evaluation with Fixed Evaluator (existing answers)")
print("=" * 60)
print(f"Normal RAG:   {old_n_acc:.1f}% -> {new_n_acc:.1f}%  (delta: {new_n_acc - old_n_acc:+.1f}%)")
print(f"Trust-Aware:  {old_t_acc:.1f}% -> {new_t_acc:.1f}%  (delta: {new_t_acc - old_t_acc:+.1f}%)")
print(f"Changes:      {sum(r['changed'] for r in corrected)}/{total} cases changed")
print(f"Abstained:    {sum(r['trust_abstained'] for r in corrected)}/25")
