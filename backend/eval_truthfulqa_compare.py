import time
import json
import logging
from pathlib import Path
import numpy as np

from evaluation.datasets.truthfulqa_loader import load_truthfulqa
from evaluation.runner import run_single_case
from app.embeddings.embedding_model import get_embedding_model
from app.llm.groq_client import get_groq_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def main():
    items = load_truthfulqa(limit=25, seed=42)
    emb = get_embedding_model()
    llm = get_groq_client()

    records = []
    print(f"Starting evaluation of {len(items)} TruthfulQA cases...")
    
    for i, item in enumerate(items):
        try:
            rec = run_single_case(item, emb, llm, top_k=3, evaluation_mode="live_llm")
            records.append(rec)
            print(
                f"[{i+1:02d}/25] {item.id} | Normal={rec['normal_correct']} (hallu={rec['normal_hallucinated']}) | "
                f"Trust={rec['trust_correct']} (abs={rec['trust_abstained']}, hallu={rec['trust_hallucinated']}) | "
                f"Score={rec['trust_score']:.3f} | Decision={rec['trust_decision'] if 'trust_decision' in rec else 'N/A'}"
            )
        except Exception as exc:
            print(f"[{i+1:02d}/25] Error evaluating {item.id}: {exc}")
        time.sleep(1.2)

    norm_acc = float(np.mean([r["normal_correct"] for r in records])) * 100
    trust_acc = float(np.mean([r["trust_correct"] for r in records])) * 100
    norm_hallu = float(np.mean([r["normal_hallucinated"] for r in records])) * 100
    trust_hallu = float(np.mean([r["trust_hallucinated"] for r in records])) * 100
    norm_abs = float(np.mean([r["normal_abstained"] for r in records])) * 100
    trust_abs = float(np.mean([r["trust_abstained"] for r in records])) * 100
    mean_trust = float(np.mean([r["trust_score"] for r in records]))
    norm_lat = float(np.mean([r["normal_latency_ms"] for r in records]))
    trust_lat = float(np.mean([r["trust_latency_ms"] for r in records]))

    results = {
        "dataset": "TruthfulQA",
        "sample_count": len(records),
        "normal_rag": {
            "accuracy": round(norm_acc, 1),
            "hallucination_rate": round(norm_hallu, 1),
            "abstention_rate": round(norm_abs, 1),
            "latency_ms": round(norm_lat, 1),
        },
        "trust_aware": {
            "accuracy": round(trust_acc, 1),
            "hallucination_rate": round(trust_hallu, 1),
            "abstention_rate": round(trust_abs, 1),
            "mean_trust_score": round(mean_trust, 3),
            "latency_ms": round(trust_lat, 1),
        }
    }

    out_file = Path("evaluation/reports/truthfulqa_evaluation_summary.json")
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "="*50)
    print("TRUTHFULQA EVALUATION SUMMARY (25 SAMPLES)")
    print("="*50)
    print(json.dumps(results, indent=2))
    print("="*50)

if __name__ == "__main__":
    main()
