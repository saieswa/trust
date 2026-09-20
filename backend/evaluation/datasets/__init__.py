"""
Evaluation Datasets package.
"""

from .loaders import EvalItem, check_dataset_availability, load_all_datasets
from .haleval_loader import load_halueval
from .truthfulqa_loader import load_truthfulqa
from .fever_loader import load_fever
from .hotpotqa_loader import load_hotpotqa

__all__ = [
    "EvalItem",
    "check_dataset_availability",
    "load_all_datasets",
    "load_halueval",
    "load_truthfulqa",
    "load_fever",
    "load_hotpotqa",
]
