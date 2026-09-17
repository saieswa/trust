"""API endpoints for Baseline vs. Trust-Aware Multi-Agent RAG comparative evaluation."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter

from app.evaluation.evaluator import (
    load_stored_evaluation_results,
    run_benchmark_evaluation,
    run_comparative_evaluation,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/evaluation")


@router.get("/comparative")
def get_comparative_evaluation_results() -> dict[str, Any]:
    """Retrieve stored comparative evaluation results comparing Baseline RAG and Trust-Aware RAG."""
    return load_stored_evaluation_results()


@router.post("/run")
def execute_comparative_evaluation() -> dict[str, Any]:
    """Trigger a live evaluation run across all 10 stress test cases and return metrics."""
    return run_comparative_evaluation(persist=True)


@router.get("/comparison-report")
def get_comparison_report() -> dict[str, Any]:
    """Retrieve structured 9-point architectural comparison report for Review 2 presentation."""
    data = load_stored_evaluation_results()
    return data.get("nine_point_comparison", {})


@router.get("/benchmark")
def get_benchmark_results() -> dict[str, Any]:
    """Legacy endpoint returning benchmark metrics for backwards compatibility."""
    return run_benchmark_evaluation()
