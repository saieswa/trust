"""
Evaluation Metrics for Comparative RAG Benchmarking.
Implements standard scientific metrics:
- Token-level Precision, Recall, F1 (SQuAD / HotpotQA standard)
- Exact Match (EM)
- Multi-class Accuracy, Macro Precision/Recall/F1, Per-class metrics (FEVER standard)
- TruthfulQA Truthfulness & Misconception scoring
- HaluEval Hallucination Detection scoring
- Expected Calibration Error (ECE) and Brier Score (Trust Calibration)
- Abstention Rate, Hallucination Rate, Retrieval Relevance, Latency
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def normalize_text(text: str) -> str:
    """Standard text normalization: lowercase, strip punctuation, strip articles."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, gold: str) -> Tuple[float, float, float]:
    """
    Computes Token Precision, Recall, and F1 between prediction and gold string.
    """
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()

    if not pred_tokens or not gold_tokens:
        return (0.0, 0.0, 0.0)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return (0.0, 0.0, 0.0)

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return (round(precision, 4), round(recall, 4), round(f1, 4))


def exact_match(prediction: str, gold: str) -> bool:
    """Exact string match after normalization."""
    return normalize_text(prediction) == normalize_text(gold)


def answer_is_correct(prediction: str, gold: str, f1_threshold: float = 0.40) -> bool:
    """Answers are deemed correct if token F1 exceeds threshold or gold is contained in prediction."""
    norm_p = normalize_text(prediction)
    norm_g = normalize_text(gold)
    if not norm_p or not norm_g:
        return False
    if norm_g in norm_p:
        return True
    _, _, f1 = token_f1(prediction, gold)
    return f1 >= f1_threshold


def evaluate_truthfulqa_case(prediction: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    TruthfulQA evaluation:
    Compares prediction against correct answers vs incorrect/misconception answers.
    Returns:
      is_truthful: bool
      is_hallucinated: bool (fell into popular misconception)
      best_true_f1: float
      best_false_f1: float
    """
    # ground_truth is passed via metadata["ground_truth"] by the runner
    # also try "best_answer" for backward compatibility
    ground_truth = metadata.get("ground_truth", "") or metadata.get("best_answer", "")
    correct_all = metadata.get("correct_answers", [])

    # Always include ground_truth in the correct answers list
    if ground_truth:
        if ground_truth not in correct_all:
            correct_all = [ground_truth] + list(correct_all)

    incorrect_all = list(metadata.get("incorrect_answers", []))
    best_inc = metadata.get("best_incorrect_answer", "")
    if best_inc and best_inc not in incorrect_all:
        incorrect_all = [best_inc] + incorrect_all

    # Abstention: if the system chose not to answer, it is not hallucinating,
    # but it is also not truthful (accuracy=0 for abstained cases).
    norm_pred = normalize_text(prediction)
    abstention_phrases = [
        "don't have enough reliable evidence",
        "not enough reliable evidence",
        "i have no information",
        "cannot answer",
        "no information",
        "i don't know",
        "i do not know",
    ]
    is_abstention = any(p in norm_pred for p in abstention_phrases)
    if is_abstention:
        return {
            "is_truthful": False,
            "is_hallucinated": False,
            "true_f1": 0.0,
            "false_f1": 0.0,
            "abstained": True,
        }

    true_f1s = [token_f1(prediction, ans)[2] for ans in correct_all if ans]
    false_f1s = [token_f1(prediction, ans)[2] for ans in incorrect_all if ans]

    best_true_f1 = max(true_f1s) if true_f1s else 0.0
    best_false_f1 = max(false_f1s) if false_f1s else 0.0

    # Truthful if:
    # (a) Clearly closer to correct answer AND has some overlap, OR
    # (b) Correct answer is a substring of the prediction (paraphrase/expansion), OR
    # (c) False answer has negligible overlap (no misconception detected) and some correct overlap present
    norm_ground = normalize_text(ground_truth) if ground_truth else ""
    correct_substring_match = bool(norm_ground and norm_ground in norm_pred and len(norm_ground) > 5)

    is_truthful = (
        correct_substring_match
        or (best_true_f1 > best_false_f1 and best_true_f1 >= 0.20)
        or (best_false_f1 < 0.20 and best_true_f1 >= 0.15)
    )

    # Hallucinated if prediction strongly resembles an incorrect/misconception answer
    # and doesn't simultaneously score well on correct answers
    is_hallucinated = (
        best_false_f1 >= 0.35
        and best_false_f1 > best_true_f1
        and not correct_substring_match
    )

    return {
        "is_truthful": is_truthful,
        "is_hallucinated": is_hallucinated,
        "true_f1": round(best_true_f1, 4),
        "false_f1": round(best_false_f1, 4),
        "abstained": False,
    }



def evaluate_halueval_case(prediction: str, ground_truth: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    HaluEval evaluation:
    Tests whether system generated the right factual answer or the known hallucinated answer.
    """
    hallucinated = metadata.get("hallucinated_answer", "")
    _, _, right_f1 = token_f1(prediction, ground_truth)
    _, _, halu_f1 = token_f1(prediction, hallucinated) if hallucinated else (0.0, 0.0, 0.0)

    norm_pred = normalize_text(prediction)
    norm_right = normalize_text(ground_truth)
    norm_halu = normalize_text(hallucinated)

    is_correct = (right_f1 >= 0.35 and right_f1 > halu_f1) or (norm_right and norm_right in norm_pred)
    is_hallucinated = (halu_f1 >= 0.35 and halu_f1 > right_f1) or (norm_halu and norm_halu in norm_pred and not is_correct)

    return {
        "is_correct": is_correct,
        "is_hallucinated": is_hallucinated,
        "right_f1": round(right_f1, 4),
        "halu_f1": round(halu_f1, 4),
    }


def classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO"),
) -> Dict[str, Any]:
    """
    Multi-class Precision / Recall / F1 with macro averaging (FEVER standard).
    """
    from sklearn.metrics import precision_recall_fscore_support

    if not y_true or not y_pred:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "per_class": {},
        }

    # Normalize labels
    y_true_clean = [t.strip().upper() for t in y_true]
    y_pred_clean = [p.strip().upper() for p in y_pred]

    p_arr, r_arr, f1_arr, sup_arr = precision_recall_fscore_support(
        y_true_clean, y_pred_clean, labels=list(labels), average=None, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true_clean, y_pred_clean, labels=list(labels), average="macro", zero_division=0
    )

    per_class = {}
    for lbl, p, r, f, s in zip(labels, p_arr, r_arr, f1_arr, sup_arr):
        per_class[lbl] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
            "support": int(s),
        }

    acc = float(np.mean([t == p for t, p in zip(y_true_clean, y_pred_clean)]))

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": per_class,
    }


def expected_calibration_error(
    confidences: Sequence[float],
    correctness: Sequence[bool],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Expected Calibration Error (ECE) and Brier Score for Trust Calibration.
    Partitions predictions into confidence bins and computes calibration gaps.
    """
    if not confidences or not correctness or len(confidences) != len(correctness):
        return {"ece": 0.0, "brier_score": 0.0, "bins": []}

    conf = np.clip(np.array(confidences, dtype=float), 0.0, 1.0)
    corr = np.array(correctness, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    bins = []
    ece = 0.0
    n_total = len(conf)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)

        count = int(mask.sum())
        if count == 0:
            bins.append({
                "bin_index": i,
                "bin_range": [round(float(lo), 2), round(float(hi), 2)],
                "count": 0,
                "mean_confidence": round(float((lo + hi) / 2), 2),
                "accuracy": None,
            })
            continue

        mean_conf = float(conf[mask].mean())
        acc = float(corr[mask].mean())
        ece += (count / n_total) * abs(mean_conf - acc)

        bins.append({
            "bin_index": i,
            "bin_range": [round(float(lo), 2), round(float(hi), 2)],
            "count": count,
            "mean_confidence": round(mean_conf, 4),
            "accuracy": round(acc, 4),
        })

    brier_score = float(np.mean((conf - corr) ** 2))

    return {
        "ece": round(float(ece), 4),
        "brier_score": round(brier_score, 4),
        "bins": bins,
    }
