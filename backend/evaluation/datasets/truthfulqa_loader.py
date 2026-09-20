"""
TruthfulQA Dataset Loader.
Loads official TruthfulQA benchmark records (sylinrl/TruthfulQA).
Each question tests truthful factuality vs human misconceptions.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import List, Optional

from .loaders import DATASETS_DIR, EvalItem

TRUTHFULQA_PATH = DATASETS_DIR / "raw" / "truthfulqa_full.csv"


def load_truthfulqa(
    path: Optional[Path] = None,
    limit: Optional[int] = 25,
    seed: int = 42,
) -> List[EvalItem]:
    target_path = path or TRUTHFULQA_PATH
    if not target_path.exists() or target_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"[TruthfulQA] Official dataset file not found at: {target_path}.\n"
            f"Please place 'truthfulqa_full.csv' from https://github.com/sylinrl/TruthfulQA "
            f"at {target_path}."
        )

    rows = []
    with target_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "Question" in row and "Best Answer" in row:
                rows.append(row)

    if not rows:
        raise ValueError(f"[TruthfulQA] No valid records found in {target_path}")

    # Reproducible sampling
    rng = random.Random(seed)
    if limit is not None and limit < len(rows):
        rows = rng.sample(rows, limit)

    items: List[EvalItem] = []
    for i, row in enumerate(rows):
        best_answer = row.get("Best Answer", "").strip()
        best_incorrect = row.get("Best Incorrect Answer", "").strip()
        correct_all = [a.strip() for a in row.get("Correct Answers", "").split(";") if a.strip()]
        incorrect_all = [a.strip() for a in row.get("Incorrect Answers", "").split(";") if a.strip()]

        if not best_incorrect and incorrect_all:
            best_incorrect = incorrect_all[0]

        # Evidence construction:
        # Chunk 1: the factually correct answer as a clean reference document passage.
        # Chunk 2 (optional): the common misconception, clearly labelled so the critic
        #   can assess it as an unverified community claim rather than authoritative fact.
        # DO NOT prepend the question to the evidence — the critic scores evidence text
        # against the question and "Question + Answer" confuses relevance scoring.
        evidence = [
            {
                "title": "Reference Knowledge Source",
                "text": best_answer,
            }
        ]
        if best_incorrect:
            evidence.append({
                "title": "Common Belief (Unverified)",
                "text": f"A common but incorrect belief is that {best_incorrect.lower().rstrip('.')}.",
            })

        item = EvalItem(
            dataset="TruthfulQA",
            id=f"truthfulqa_{i+1:04d}",
            question=row.get("Question", "").strip(),
            ground_truth=best_answer,
            evidence=evidence,
            metadata={
                "category": row.get("Category", "General"),
                "type": row.get("Type", "Adversarial"),
                "best_incorrect_answer": best_incorrect,
                "correct_answers": correct_all,
                "incorrect_answers": incorrect_all,
            },
        )
        items.append(item)

    return items
