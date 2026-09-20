"""
Unified Evaluation Item Schema and Registry.
Standardizes all datasets (HaluEval, TruthfulQA, FEVER, HotpotQA)
into a unified schema preserving original questions, evidence, and ground truth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

DATASETS_DIR = Path(__file__).parent


@dataclass
class EvalItem:
    dataset: str                     # "HaluEval" | "TruthfulQA" | "FEVER" | "HotpotQA"
    id: str                          # Unique case identifier
    question: str                    # Original question or claim
    ground_truth: str                # Target answer or label ("SUPPORTS", "REFUTES", etc.)
    evidence: List[Dict[str, str]]   # List of {"title": str, "text": str}
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def context_documents(self) -> List[Dict[str, str]]:
        """Alias for evidence compatibility with retriever indexer."""
        return self.evidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "id": self.id,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "evidence": self.evidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvalItem:
        evidence = data.get("evidence") or data.get("context_documents") or []
        return cls(
            dataset=data.get("dataset", "Unknown"),
            id=data.get("id", ""),
            question=data.get("question", ""),
            ground_truth=data.get("ground_truth", ""),
            evidence=evidence,
            metadata=data.get("metadata", {}),
        )


def check_dataset_availability() -> Dict[str, Dict[str, Any]]:
    """
    Checks existence and size of official/local dataset files.
    Returns clear status and expected local path for user if unavailable.
    """
    from .haleval_loader import HALUEVAL_PATH
    from .truthfulqa_loader import TRUTHFULQA_PATH
    from .fever_loader import FEVER_PATH
    from .hotpotqa_loader import HOTPOTQA_PATH

    checks = {
        "HaluEval": {
            "path": str(HALUEVAL_PATH),
            "available": HALUEVAL_PATH.exists() and HALUEVAL_PATH.stat().st_size > 0,
            "size_bytes": HALUEVAL_PATH.stat().st_size if HALUEVAL_PATH.exists() else 0,
            "expected_format": "JSONL/JSON array from RUCAIBox/HaluEval (qa_data.json)",
        },
        "TruthfulQA": {
            "path": str(TRUTHFULQA_PATH),
            "available": TRUTHFULQA_PATH.exists() and TRUTHFULQA_PATH.stat().st_size > 0,
            "size_bytes": TRUTHFULQA_PATH.stat().st_size if TRUTHFULQA_PATH.exists() else 0,
            "expected_format": "Official TruthfulQA.csv (790 rows)",
        },
        "FEVER": {
            "path": str(FEVER_PATH),
            "available": FEVER_PATH.exists() and FEVER_PATH.stat().st_size > 0,
            "size_bytes": FEVER_PATH.stat().st_size if FEVER_PATH.exists() else 0,
            "expected_format": "JSON/JSONL with {id, claim, label, evidence_text}",
        },
        "HotpotQA": {
            "path": str(HOTPOTQA_PATH),
            "available": HOTPOTQA_PATH.exists() and HOTPOTQA_PATH.stat().st_size > 0,
            "size_bytes": HOTPOTQA_PATH.stat().st_size if HOTPOTQA_PATH.exists() else 0,
            "expected_format": "JSON/JSONL with {id, question, answer, supporting_docs}",
        },
    }
    return checks


def load_all_datasets(
    limits: Optional[Dict[str, int]] = None,
    seed: int = 42,
) -> Dict[str, List[EvalItem]]:
    """
    Loads normalized items from all 4 datasets according to configurable limits.
    """
    from .haleval_loader import load_halueval
    from .truthfulqa_loader import load_truthfulqa
    from .fever_loader import load_fever
    from .hotpotqa_loader import load_hotpotqa

    limits = limits or {}
    return {
        "HaluEval": load_halueval(limit=limits.get("HaluEval", 25), seed=seed),
        "TruthfulQA": load_truthfulqa(limit=limits.get("TruthfulQA", 25), seed=seed),
        "FEVER": load_fever(limit=limits.get("FEVER", 20), seed=seed),
        "HotpotQA": load_hotpotqa(limit=limits.get("HotpotQA", 20), seed=seed),
    }
