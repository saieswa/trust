"""API endpoints for Critic Agent evaluation."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.agents.critic import get_critic_agent
from app.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/critic")


class CriticRequest(BaseModel):
    document_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    evidence: list[dict[str, Any]] | None = None


_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


@router.post("/evaluate")
def evaluate_evidence(request: CriticRequest) -> dict[str, Any]:
    """Evaluate retrieved evidence chunks for a question and target document."""
    from app.api.validation import validate_document_id
    from app.llm.groq_client import GroqClientError

    document_id = validate_document_id(request.document_id)
    question = (request.question or "").strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="question must contain non-empty text",
        )

    evidence = request.evidence
    if evidence is None:
        try:
            evidence = get_retriever().retrieve(
                question=question,
                document_id=document_id,
            )
        except Exception as exc:
            logger.exception(
                "Retrieval for critic failed document_id=%s query=%s",
                document_id,
                question,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Evidence retrieval is unavailable.",
            ) from exc

    try:
        result = get_critic_agent().evaluate(
            question=question,
            evidence=evidence,
            document_id=document_id,
        )
        return result
    except Exception as exc:
        logger.exception("Critic evaluation failed document_id=%s query=%s", document_id, question)
        err_msg = "Unable to connect to the language model." if isinstance(exc, GroqClientError) else "Document could not be processed."
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=err_msg,
        ) from exc


@router.get("/contradictions/{document_id}")
def get_document_contradictions(document_id: str) -> dict[str, Any]:
    """Retrieve persisted contradiction history for a document."""
    from app.api.validation import validate_document_id
    from app.database.metadata import get_contradictions
    doc_id = validate_document_id(document_id)
    return {
        "document_id": doc_id,
        "contradictions": get_contradictions(doc_id),
    }


class ContradictionCheckRequest(BaseModel):
    chunk_a: str = Field(min_length=1)
    chunk_b: str = Field(min_length=1)
    chunk_a_id: str | None = "chunk-a"
    chunk_b_id: str | None = "chunk-b"


@router.post("/contradictions/detect")
def detect_pair_contradiction(request: ContradictionCheckRequest) -> dict[str, Any]:
    """Directly evaluate whether two statements contradict each other."""
    from app.agents.contradiction_detector import get_contradiction_detector
    detector = get_contradiction_detector()
    res = detector.compare_pair(
        chunk_a_id=request.chunk_a_id or "chunk-a",
        chunk_a_text=request.chunk_a,
        chunk_b_id=request.chunk_b_id or "chunk-b",
        chunk_b_text=request.chunk_b,
    )
    return res.to_dict()

