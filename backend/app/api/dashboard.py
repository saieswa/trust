"""API endpoints for Trust-Aware RAG system dashboard and real metrics."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Query

from app.database.query_store import get_actual_documents, get_system_metrics

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/dashboard")


@router.get("/stats")
def get_dashboard_statistics(
    document_id: str | None = Query(default=None, description="Optional document ID filter"),
    start_date: str | None = Query(default=None, description="Optional start ISO date"),
    end_date: str | None = Query(default=None, description="Optional end ISO date"),
) -> dict[str, Any]:
    """Retrieve actual stored system metrics without synthetic or hardcoded values."""
    from app.api.validation import validate_document_id
    clean_doc_id = validate_document_id(document_id) if document_id else None
    try:
        return get_system_metrics(
            document_id=clean_doc_id,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as exc:
        logger.exception("Failed computing dashboard statistics: %s", exc)
        return {
            "has_data": False,
            "total_documents": 0,
            "total_questions": 0,
            "average_trust_score": None,
            "high_trust_responses": 0,
            "medium_trust_responses": 0,
            "low_trust_abstained_responses": 0,
            "verified_answers": 0,
            "unsupported_answers": 0,
            "contradictions_detected": 0,
            "average_retrieval_relevance": None,
            "average_response_time_ms": None,
            "documents": [],
            "queries": [],
            "trust_distribution": [],
            "verification_distribution": [],
            "timeline": [],
            "error": "Failed to compute dashboard metrics.",
        }


@router.get("/documents")
def get_dashboard_documents() -> list[dict[str, Any]]:
    """Retrieve registered stored documents for filtering."""
    try:
        return get_actual_documents()
    except Exception as exc:
        logger.exception("Failed fetching dashboard documents: %s", exc)
        return []
