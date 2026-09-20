"""
HotpotQA Dataset Loader.
Evaluates multi-hop question answering across multiple context paragraphs.
Schema:
  {"id": str, "question": str, "answer": str, "supporting_docs": [{"title": str, "text": str}]}
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional

from .loaders import DATASETS_DIR, EvalItem

HOTPOTQA_PATH = DATASETS_DIR / "hotpotqa_subset.json"


def load_hotpotqa(
    path: Optional[Path] = None,
    limit: Optional[int] = 20,
    seed: int = 42,
) -> List[EvalItem]:
    target_path = path or HOTPOTQA_PATH
    if not target_path.exists() or target_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"[HotpotQA] Dataset file not found at: {target_path}.\n"
            f"Please place 'hotpotqa_subset.json' or official 'hotpot_dev_distractor_v1.json' from "
            f"https://hotpotqa.github.io at {target_path}."
        )

    records = []
    with target_path.open("r", encoding="utf-8") as f:
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
        raise ValueError(f"[HotpotQA] No valid records found in {target_path}")

    rng = random.Random(seed)
    if limit is not None and limit < len(records):
        records = rng.sample(records, limit)

    items: List[EvalItem] = []
    for i, rec in enumerate(records):
        q = rec.get("question", "").strip()
        ans = rec.get("answer", "").strip()

        evidence = []
        raw_docs = rec.get("supporting_docs") or rec.get("context") or []
        for doc in raw_docs:
            if isinstance(doc, dict):
                evidence.append({
                    "title": doc.get("title", "Document"),
                    "text": doc.get("text", "") if isinstance(doc.get("text"), str) else "".join(doc.get("text", [])),
                })
            elif isinstance(doc, (list, tuple)) and len(doc) >= 2:
                # HotpotQA native dev format: [title, [sentences...]]
                title = doc[0]
                text = " ".join(doc[1]) if isinstance(doc[1], list) else str(doc[1])
                evidence.append({"title": title, "text": text})

        item = EvalItem(
            dataset="HotpotQA",
            id=rec.get("id", f"hotpot_{i+1:04d}"),
            question=q,
            ground_truth=ans,
            evidence=evidence,
            metadata={
                "type": rec.get("type", "multi-hop"),
                "level": rec.get("level", "medium"),
            },
        )
        items.append(item)

    return items
