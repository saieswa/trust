"""
FEVER (Fact Extraction and VERification) Dataset Loader.
Evaluates claim verification into three classes: SUPPORTS, REFUTES, NOT ENOUGH INFO.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional

from .loaders import DATASETS_DIR, EvalItem

FEVER_PATH = DATASETS_DIR / "fever_subset.json"


def load_fever(
    path: Optional[Path] = None,
    limit: Optional[int] = 20,
    seed: int = 42,
) -> List[EvalItem]:
    target_path = path or FEVER_PATH
    if not target_path.exists() or target_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"[FEVER] Dataset file not found at: {target_path}.\n"
            f"Please place 'fever_subset.json' or official 'shared_task_dev.jsonl' from "
            f"https://fever.ai at {target_path}."
        )

    records = []
    with target_path.open("r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            raw_items = json.load(f)
            for item in raw_items:
                if isinstance(item, dict) and "claim" in item:
                    records.append(item)
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if isinstance(item, dict) and "claim" in item:
                        records.append(item)
                except json.JSONDecodeError:
                    continue

    if not records:
        raise ValueError(f"[FEVER] No valid records found in {target_path}")

    rng = random.Random(seed)
    if limit is not None and limit < len(records):
        records = rng.sample(records, limit)

    items: List[EvalItem] = []
    for i, rec in enumerate(records):
        claim = rec.get("claim", "").strip()
        label = rec.get("label", "NOT ENOUGH INFO").strip().upper()
        ev_text = rec.get("evidence_text", "").strip()
        ev_source = rec.get("evidence_source", "Wikipedia Source").strip()

        evidence = []
        if ev_text:
            evidence.append({
                "title": ev_source,
                "text": ev_text,
            })

        item = EvalItem(
            dataset="FEVER",
            id=rec.get("id", f"fever_{i+1:04d}"),
            question=claim,
            ground_truth=label,
            evidence=evidence,
            metadata={
                "claim": claim,
                "label": label,
                "evidence_source": ev_source,
            },
        )
        items.append(item)

    return items
