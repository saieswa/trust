"""Abstention Engine for Trust-Aware RAG.

Formulates transparent, honest refusals when evidence is insufficient,
contradictory, or falls below trust thresholds. Ensures the system never
hallucinates when trustworthy ground-truth evidence is unavailable.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from app.trust.decision import AbstentionReason, TrustDecision
from app.trust.trust_model import TrustScoreResult


class AbstentionEngine:
    """Agent responsible for constructing grounded abstention responses."""

    def build_abstention_response(
        self,
        question: str,
        document_id: str,
        decision: TrustDecision,
        trust_result: TrustScoreResult,
        evidence: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Construct a structured abstention response explaining why the system abstained."""
        reason = decision.abstention_reason or AbstentionReason.INSUFFICIENT_EVIDENCE
        explanation_lines = []

        if reason == AbstentionReason.CONTRADICTION_DETECTED:
            statement = (
                "I cannot answer this question because conflicting factual statements were "
                f"detected within the document passages ({trust_result.factors.contradicting_chunks} contradiction(s))."
            )
            details = "To prevent hallucination, the Trust-Aware framework abstains when evidence contradicts itself."
        elif reason == AbstentionReason.CROSS_DOCUMENT_MISMATCH:
            statement = (
                f"No verified evidence belonging to document '{document_id}' was available to answer this question."
            )
            details = "Cross-document retrieval is strictly blocked to maintain strict document isolation."
        elif reason == AbstentionReason.LOW_TRUST_SCORE:
            statement = "I don't have enough reliable evidence in the uploaded document to answer this question."
            details = f"The evidence trust score ({trust_result.overall_score:.1%}) is below the acceptable threshold."
        else:
            statement = "I don't have enough reliable evidence in the uploaded document to answer this question."
            details = "The retrieved passages do not contain relevant information addressing this query."

        full_message = f"## Abstention Notice\n{statement}\n\n**Reason:** {decision.reason}\n\n**Safety Note:** {details}"

        return {
            "answer": full_message,
            "abstention": True,
            "abstention_reason": reason.value if hasattr(reason, "value") else str(reason),
            "document_id": document_id,
            "question": question,
            "trust_score": trust_result.to_dict(),
            "decision": decision.to_dict(),
            "sources": [],
            "evidence": list(evidence or []),
            "retrieval_results": list(evidence or []),
            "claims": [],
            "verification": {
                "status": "ABSTAINED",
                "hallucination_risk": "ZERO_HALLUCINATION_VIA_ABSTENTION",
                "verified_claims_count": 0,
                "total_claims_count": 0,
            },
        }


_abstention_engine = AbstentionEngine()


def get_abstention_engine() -> AbstentionEngine:
    return _abstention_engine
