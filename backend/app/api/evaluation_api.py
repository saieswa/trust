"""API endpoints for Baseline vs. Trust-Aware Multi-Agent RAG comparative evaluation."""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from app.evaluation.evaluator import (
    load_stored_evaluation_results,
    run_benchmark_evaluation,
    run_comparative_evaluation,
)
from evaluation.runner import (
    check_llm_status,
    get_latest_results,
    run_full_evaluation,
)
from evaluation.datasets.loaders import check_dataset_availability

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/evaluation")

_EVAL_STATE: Dict[str, Any] = {
    "is_running": False,
    "step": "IDLE",
    "progress": 0,
    "message": "Ready to run dataset evaluation",
    "error": None,
    "last_completed": None,
}
_EVAL_LOCK = threading.Lock()


class DatasetRunRequest(BaseModel):
    limits: Optional[Dict[str, int]] = Field(
        default=None,
        description="Case limits per dataset, e.g. {'HaluEval': 25, 'TruthfulQA': 25, 'FEVER': 20, 'HotpotQA': 20}",
    )
    seed: int = Field(default=42, description="Random sampling seed for reproducible evaluation")
    top_k: int = Field(default=3, description="Top-k evidence chunks retrieved per case")


@router.get("/dataset/status")
def get_dataset_evaluation_status() -> Dict[str, Any]:
    """
    Returns dataset availability, file sizes, expected paths, LLM availability status,
    and whether an evaluation run is currently active.
    """
    datasets_avail = check_dataset_availability()
    llm_avail, llm_reason = check_llm_status()

    return {
        "datasets": datasets_avail,
        "llm": {
            "available": llm_avail,
            "status": llm_reason,
            "mode": "live_llm" if llm_avail else "deterministic_fallback",
        },
        "runner_state": _EVAL_STATE,
    }


@router.get("/dataset/results")
def get_dataset_evaluation_results() -> Dict[str, Any]:
    """
    Returns latest aggregated benchmark metrics across HaluEval, TruthfulQA, FEVER, and HotpotQA.
    If no run exists yet, runs an initial baseline evaluation.
    """
    latest = get_latest_results()
    if latest is None:
        # Run a small baseline if no evaluation has been saved yet
        try:
            latest = run_full_evaluation(
                limits={"HaluEval": 5, "TruthfulQA": 5, "FEVER": 5, "HotpotQA": 5},
                seed=42,
            )
        except Exception as exc:
            logger.exception("Initial dataset evaluation failed: %s", exc)
            raise HTTPException(
                status_code=500,
                detail=f"No existing evaluation results and auto-run failed: {exc}",
            )
    return latest


def _background_eval_task(limits: Optional[Dict[str, int]], seed: int, top_k: int):
    global _EVAL_STATE

    def progress_callback(update: Dict[str, Any]):
        global _EVAL_STATE
        _EVAL_STATE["step"] = update.get("step", "EVALUATING")
        _EVAL_STATE["progress"] = update.get("progress", _EVAL_STATE["progress"])
        _EVAL_STATE["message"] = update.get("message", "")

    try:
        results = run_full_evaluation(
            limits=limits,
            seed=seed,
            top_k=top_k,
            progress_callback=progress_callback,
        )
        with _EVAL_LOCK:
            _EVAL_STATE["is_running"] = False
            _EVAL_STATE["step"] = "COMPLETE"
            _EVAL_STATE["progress"] = 100
            _EVAL_STATE["message"] = f"Evaluation completed ({results['evaluation_metadata']['total_cases']} cases)."
            _EVAL_STATE["last_completed"] = results["evaluation_metadata"]["timestamp"]
            _EVAL_STATE["error"] = None
    except Exception as exc:
        logger.exception("Dataset evaluation background run failed: %s", exc)
        with _EVAL_LOCK:
            _EVAL_STATE["is_running"] = False
            _EVAL_STATE["step"] = "FAILED"
            _EVAL_STATE["message"] = f"Evaluation failed: {exc}"
            _EVAL_STATE["error"] = str(exc)


@router.post("/dataset/run")
def trigger_dataset_evaluation(
    request: DatasetRunRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    """
    Triggers a real dataset evaluation run across HaluEval, TruthfulQA, FEVER, and HotpotQA.
    Executes both Normal RAG and Trust-Aware RAG on identical items and calculates metrics.
    """
    global _EVAL_STATE
    with _EVAL_LOCK:
        if _EVAL_STATE["is_running"]:
            return {
                "status": "already_running",
                "message": "An evaluation run is already in progress.",
                "runner_state": _EVAL_STATE,
            }
        _EVAL_STATE["is_running"] = True
        _EVAL_STATE["step"] = "STARTING"
        _EVAL_STATE["progress"] = 5
        _EVAL_STATE["message"] = "Initializing evaluation environment..."
        _EVAL_STATE["error"] = None

    background_tasks.add_task(
        _background_eval_task,
        limits=request.limits,
        seed=request.seed,
        top_k=request.top_k,
    )

    return {
        "status": "started",
        "message": "Dataset evaluation pipeline started in background.",
        "runner_state": _EVAL_STATE,
    }


@router.get("/dataset/progress")
def get_evaluation_progress() -> Dict[str, Any]:
    """Returns the live progress of the currently executing evaluation run."""
    return _EVAL_STATE


# ==============================================================================
# DETERMINISTIC STRESS TESTS (PRESERVED OLD 10-CASE BENCHMARK)
# ==============================================================================

@router.get("/comparative")
def get_comparative_evaluation_results() -> dict[str, Any]:
    """Retrieve stored comparative evaluation results from deterministic stress tests."""
    return load_stored_evaluation_results()


@router.post("/run")
def execute_comparative_evaluation() -> dict[str, Any]:
    """Trigger a live evaluation run across all 10 deterministic stress test cases."""
    return run_comparative_evaluation(persist=True)


@router.get("/comparison-report")
def get_comparison_report() -> dict[str, Any]:
    """Retrieve structured comparison report for defense presentation."""
    data = load_stored_evaluation_results()
    return data.get("nine_point_comparison", {})


@router.get("/benchmark")
def get_benchmark_results() -> dict[str, Any]:
    """Legacy endpoint returning benchmark metrics for backwards compatibility."""
    return run_benchmark_evaluation()
