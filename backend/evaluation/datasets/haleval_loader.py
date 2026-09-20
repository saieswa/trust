"""
HaluEval Dataset Loader.
Loads official HaluEval QA benchmark records from RUCAIBox/HaluEval.
Schema:
  {"knowledge": str, "question": str, "right_answer": str, "hallucinated_answer": str}
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional

from .loaders import DATASETS_DIR, EvalItem

HALUEVAL_PATH = DATASETS_DIR / "raw" / "halueval_qa_data.json"


def load_halueval(
    path: Optional[Path] = None,
    limit: Optional[int] = 25,
    seed: int = 42,
) -> List[EvalItem]:
    target_path = path or HALUEVAL_PATH
    if not target_path.exists() or target_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"[HaluEval] Official dataset file not found at: {target_path}.\n"
            f"Please download the official 'qa_data.json' from "
            f"https://github.com/RUCAIBox/HaluEval/blob/main/data/qa_data.json "
            f"and place it at {target_path}."
        )

    records = []
    with target_path.open("r", encoding="utf-8") as f:
        # Check first non-whitespace character to handle both JSONL and JSON array
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            raw_items = json.load(f)
            for item in raw_items:
                if isinstance(item, dict) and "question" in item:
                    records.append(item)
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if isinstance(item, dict) and "question" in item:
                        records.append(item)
                except json.JSONDecodeError:
                    continue

    if not records:
        raise ValueError(f"[HaluEval] No valid records found in {target_path}")

    # Reproducible sampling
    rng = random.Random(seed)
    if limit is not None and limit < len(records):
        records = rng.sample(records, limit)

    items: List[EvalItem] = []
    for i, rec in enumerate(records):
        q = rec.get("question", "").strip()
        gt = rec.get("right_answer", "").strip()
        knowledge = rec.get("knowledge", "").strip()
        hallucinated = rec.get("hallucinated_answer", "").strip()

        evidence = []
        if knowledge:
            evidence.append({
                "title": "Knowledge Document",
                "text": knowledge,
            })

        item = EvalItem(
            dataset="HaluEval",
            id=f"halueval_{i+1:04d}",
            question=q,
            ground_truth=gt,
            evidence=evidence,
            metadata={
                "hallucinated_answer": hallucinated,
                "domain": "QA",
                "raw_keys": list(rec.keys()),
            },
        )
        items.append(item)

    return items
