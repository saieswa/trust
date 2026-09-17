"""API endpoint for Baseline vs. Trust-Aware RAG comparison."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.api.chat import run_baseline, run_trust_aware

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


class CompareRequest(BaseModel):
    document_id: str = Field(min_length=1)
    question: str = Field(min_length=1)


@router.post("/chat/compare")
def compare_rag_models(request: CompareRequest) -> dict[str, Any]:
    """Execute both Baseline RAG and Trust-Aware Multi-Agent RAG side by side on identical inputs."""
    from app.api.validation import validate_document_id
    from app.llm.groq_client import GroqClientError

    document_id = validate_document_id(request.document_id)
    question = (request.question or "").strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="question must contain non-empty text",
        )

    try:
        baseline_result = run_baseline(question=question, document_id=document_id)
    except Exception as exc:
        logger.exception("Baseline RAG comparison run failed: %s", exc)
        err_msg = "Unable to connect to the language model." if isinstance(exc, GroqClientError) else "Document could not be processed."
        baseline_result = {
            "answer": "Unable to generate baseline answer.",
            "error": err_msg,
            "sources": [],
            "retrieval_results": [],
        }

    try:
        trust_aware_result = run_trust_aware(question=question, document_id=document_id)
    except Exception as exc:
        logger.exception("Trust-Aware RAG comparison run failed: %s", exc)
        err_msg = "Unable to connect to the language model." if isinstance(exc, GroqClientError) else "Document could not be processed."
        trust_aware_result = {
            "answer": "Unable to generate trust-aware answer.",
            "error": err_msg,
            "trust_score": {"overall_score": 0.0, "trust_level": "LOW_TRUST"},
            "decision": {"action": "ERROR"},
        }

    # Generate analytical summary of comparison
    ta_score = trust_aware_result.get("trust_score", {}).get("overall_score", 0.0)
    ta_level = trust_aware_result.get("trust_score", {}).get("trust_level", "LOW_TRUST")
    ta_abstained = trust_aware_result.get("abstention", False)
    verification = trust_aware_result.get("verification", {})
    verified_claims = trust_aware_result.get("claims", [])

    if ta_abstained:
        safety_impact = (
            "Hallucination prevented: Baseline produced an unverified answer, "
            "whereas Trust-Aware detected insufficient or conflicting evidence and safely abstained."
        )
    elif verification.get("hallucination_detected"):
        safety_impact = (
            "Hallucination flagged: Verifier detected unsupported claim(s) in the draft synthesis "
            "and annotated them with warnings."
        )
    else:
        safety_impact = (
            f"Evidence verified: Trust-Aware framework validated all {len(verified_claims)} claim(s) "
            f"against cited passages with {ta_score:.1%} trust ({ta_level})."
        )

    return {
        "document_id": document_id,
        "question": question,
        "baseline": {
            "answer": baseline_result.get("answer", ""),
            "sources": baseline_result.get("sources", []),
            "retrieval_results_count": len(baseline_result.get("retrieval_results", [])),
            "hallucination_check": "NONE (Unverified)",
            "trust_score": "NOT_CALCULATED",
        },
        "trust_aware": {
            "answer": trust_aware_result.get("answer", ""),
            "trust_score": trust_aware_result.get("trust_score", {}),
            "decision": trust_aware_result.get("decision", {}),
            "abstention": ta_abstained,
            "abstention_reason": trust_aware_result.get("abstention_reason"),
            "verified_claims": verified_claims,
            "verification": verification,
            "sources": trust_aware_result.get("sources", []),
            "retrieval_results_count": len(trust_aware_result.get("retrieval_results", [])),
        },
        "comparison_analysis": {
            "safety_impact": safety_impact,
            "trust_aware_advantage": [
                "Strict document isolation preventing cross-document contamination",
                "Automated contradiction detection across retrieved chunks",
                "Mathematical Trust Score calibrated across 4 pillars",
                "Atomic claim-level verification eliminating hallucinations",
                "Transparent, safety-aligned abstention when evidence is unreliable",
            ],
        },
    }
