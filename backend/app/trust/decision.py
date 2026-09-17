"""Trust-based Decision System for Trust-Aware RAG.

Determines the deterministic action to take based on the Trust Score:
1. HIGH (trust_score >= 0.75):
   → Continue to Synthesizer (DIRECT_ANSWER / CONTINUE_TO_SYNTHESIZER)
2. MEDIUM (0.50 <= trust_score < 0.75):
   → Retrieve additional evidence (RETRIEVE_MORE)
   → Re-run Critic
   → Recalculate Trust Score
   → Limit retrieval iterations (prevent infinite loops)
   → If still insufficient, abstain
3. LOW (trust_score < 0.50):
   → Do not generate a confident answer
   → Return an abstention response (ABSTAIN)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from app.trust.trust_model import (
    DEFAULT_CONFIGURABLE_HIGH_THRESHOLD,
    DEFAULT_CONFIGURABLE_LOW_THRESHOLD,
    TrustScoreResult,
)

DEFAULT_HIGH_THRESHOLD: float = DEFAULT_CONFIGURABLE_HIGH_THRESHOLD  # 0.75
DEFAULT_LOW_THRESHOLD: float = DEFAULT_CONFIGURABLE_LOW_THRESHOLD    # 0.50
DEFAULT_MAX_RETRIEVAL_ATTEMPTS: int = 2


class ActionType(str, Enum):
    DIRECT_ANSWER = "DIRECT_ANSWER"
    RETRIEVE_MORE = "RETRIEVE_MORE"
    ABSTAIN = "ABSTAIN"


class AbstentionReason(str, Enum):
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONTRADICTION_DETECTED = "contradiction_detected"
    LOW_TRUST_SCORE = "low_trust_score"
    CROSS_DOCUMENT_MISMATCH = "cross_document_mismatch"


@dataclass(frozen=True)
class TrustDecision:
    """Represents the deterministic routing decision of the trust system."""
    action: ActionType
    reason: str
    trust_score: float
    trust_level: str
    can_retry: bool
    number_of_retrieval_attempts: int = 1
    decision: str = ""
    abstention_reason: AbstentionReason | None = None
    high_threshold: float = DEFAULT_HIGH_THRESHOLD
    low_threshold: float = DEFAULT_LOW_THRESHOLD

    def __post_init__(self):
        if not self.decision:
            if self.action == ActionType.DIRECT_ANSWER:
                object.__setattr__(self, "decision", "CONTINUE_TO_SYNTHESIZER")
            else:
                object.__setattr__(self, "decision", self.action.value)

    def to_dict(self) -> dict[str, Any]:
        """Return structured dictionary matching user specification and backwards compatibility."""
        return {
            "trust_score": self.trust_score,
            "decision": self.decision,
            "action": self.action.value,
            "reason": self.reason,
            "number_of_retrieval_attempts": self.number_of_retrieval_attempts,
            "trust_level": self.trust_level,
            "can_retry": self.can_retry,
            "abstention_reason": self.abstention_reason.value if self.abstention_reason else None,
            "thresholds": {
                "high": self.high_threshold,
                "low": self.low_threshold,
            },
        }


def make_trust_decision(
    trust_result: TrustScoreResult,
    iteration: int = 0,
    max_iterations: int = 1,
    retrieval_attempts: int | None = None,
    max_retrieval_attempts: int | None = None,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
    low_threshold: float = DEFAULT_LOW_THRESHOLD,
) -> TrustDecision:
    """Evaluate trust score metrics against configurable thresholds and decide whether to answer, retrieve more, or abstain."""
    score = trust_result.overall_score
    factors = trust_result.factors

    # Resolve attempt counts
    attempts = retrieval_attempts if retrieval_attempts is not None else (iteration + 1)
    max_attempts = (
        max_retrieval_attempts
        if max_retrieval_attempts is not None
        else (max_iterations + 1 if max_iterations is not None else DEFAULT_MAX_RETRIEVAL_ATTEMPTS)
    )
    can_retry = attempts < max_attempts

    # 1. Check for fatal cross-chunk contradictions
    if factors.contradicting_chunks > 0:
        return TrustDecision(
            action=ActionType.ABSTAIN,
            decision="ABSTAIN",
            reason=f"Abstaining to prevent hallucination: {factors.contradicting_chunks} cross-chunk factual contradiction(s) detected.",
            trust_score=score,
            trust_level="LOW_TRUST",
            can_retry=False,
            number_of_retrieval_attempts=attempts,
            abstention_reason=AbstentionReason.CONTRADICTION_DETECTED,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
        )

    # 2. Check for all unrelated documents
    if factors.total_chunks > 0 and factors.unrelated_chunks == factors.total_chunks:
        return TrustDecision(
            action=ActionType.ABSTAIN,
            decision="ABSTAIN",
            reason="Abstaining: All retrieved evidence chunks belonged to unrelated documents. Cross-document leakage prohibited.",
            trust_score=0.0,
            trust_level="LOW_TRUST",
            can_retry=False,
            number_of_retrieval_attempts=attempts,
            abstention_reason=AbstentionReason.CROSS_DOCUMENT_MISMATCH,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
        )

    # 3. HIGH TRUST: trust_score >= high_threshold (default >= 0.75)
    if score >= high_threshold and factors.supporting_chunks >= 1:
        return TrustDecision(
            action=ActionType.DIRECT_ANSWER,
            decision="CONTINUE_TO_SYNTHESIZER",
            reason=f"High trust ({score:.1%}): {factors.relevant_chunks} relevant chunk(s) verified with high quality and no contradictions. Continuing to Synthesizer.",
            trust_score=score,
            trust_level="HIGH_TRUST",
            can_retry=False,
            number_of_retrieval_attempts=attempts,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
        )

    # 4. MEDIUM TRUST: low_threshold <= trust_score < high_threshold (default 0.50 <= score < 0.75)
    if score >= low_threshold:
        if can_retry:
            return TrustDecision(
                action=ActionType.RETRIEVE_MORE,
                decision="RETRIEVE_MORE",
                reason=f"Medium trust ({score:.1%}): Attempt {attempts}/{max_attempts}. Retrieving additional evidence to strengthen confidence.",
                trust_score=score,
                trust_level="MODERATE_TRUST",
                can_retry=True,
                number_of_retrieval_attempts=attempts,
                high_threshold=high_threshold,
                low_threshold=low_threshold,
            )
        # If retries are exhausted or maximum attempts reached -> abstain
        return TrustDecision(
            action=ActionType.ABSTAIN,
            decision="ABSTAIN",
            reason=f"Abstaining: Medium trust ({score:.1%}) remained insufficient after {attempts} retrieval iteration(s).",
            trust_score=score,
            trust_level="MODERATE_TRUST",
            can_retry=False,
            number_of_retrieval_attempts=attempts,
            abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
        )

    # 5. LOW TRUST: trust_score < low_threshold (default < 0.50)
    # Do not generate a confident answer; return an abstention response
    return TrustDecision(
        action=ActionType.ABSTAIN,
        decision="ABSTAIN",
        reason="I don't have enough reliable evidence in the uploaded document to answer this question.",
        trust_score=score,
        trust_level="LOW_TRUST",
        can_retry=False,
        number_of_retrieval_attempts=attempts,
        abstention_reason=AbstentionReason.LOW_TRUST_SCORE,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
    )
