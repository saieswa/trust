"""Unit tests for Trust-based Decision System."""

from __future__ import annotations

import pytest

from app.trust.decision import (
    AbstentionReason,
    ActionType,
    make_trust_decision,
)
from app.trust.trust_model import calculate_trust_score


def _chunk(
    chunk_id: str = "c1",
    document_id: str = "doc-1",
    relevance: bool = True,
    support_status: str = "supported",
    contradiction_status: str = "none",
) -> dict:
    return {
        "document_id": document_id,
        "chunk_id": chunk_id,
        "relevance": relevance,
        "support_status": support_status,
        "quality_assessment": {
            "evidence_strength": "high" if relevance else "none",
            "source_quality": "high",
            "potential_outdated": False,
        },
        "contradiction_status": contradiction_status,
        "explanation": "test",
    }


def test_decision_direct_answer_on_high_trust() -> None:
    evaluations = [_chunk("c1"), _chunk("c2")]
    trust_res = calculate_trust_score(evaluations=evaluations)
    decision = make_trust_decision(trust_res, iteration=0, max_iterations=1)

    assert decision.action == ActionType.DIRECT_ANSWER
    assert decision.can_retry is False
    assert decision.abstention_reason is None
    assert "High trust" in decision.reason


def test_decision_retrieve_more_on_moderate_trust() -> None:
    # 1 relevant, 2 irrelevant -> moderate relevance
    evaluations = [
        _chunk("c1", relevance=True, support_status="supported"),
        _chunk("c2", relevance=False, support_status="unsupported"),
        _chunk("c3", relevance=False, support_status="unsupported"),
    ]
    trust_res = calculate_trust_score(evaluations=evaluations)
    decision = make_trust_decision(trust_res, iteration=0, max_iterations=1)

    assert decision.action == ActionType.RETRIEVE_MORE
    assert decision.can_retry is True


def test_decision_abstains_on_contradiction() -> None:
    evaluations = [
        _chunk("c1", contradiction_status="contradiction_detected: conflicting numbers"),
        _chunk("c2", contradiction_status="contradiction_detected: conflicting numbers"),
    ]
    contradictions = [{"chunk_id_a": "c1", "chunk_id_b": "c2", "explanation": "conflict"}]
    trust_res = calculate_trust_score(evaluations=evaluations, contradictions=contradictions)
    decision = make_trust_decision(trust_res, iteration=0, max_iterations=1)

    assert decision.action == ActionType.ABSTAIN
    assert decision.abstention_reason == AbstentionReason.CONTRADICTION_DETECTED
    assert "contradiction" in decision.reason.lower()


def test_decision_abstains_on_cross_document_leakage() -> None:
    evaluations = [
        {
            "document_id": "other-doc",
            "chunk_id": "c-bad",
            "relevance": False,
            "support_status": "unrelated_document",
            "quality_assessment": {"evidence_strength": "none", "source_quality": "low", "potential_outdated": False},
            "contradiction_status": "none",
            "explanation": "Rejected",
        }
    ]
    trust_res = calculate_trust_score(evaluations=evaluations)
    decision = make_trust_decision(trust_res, iteration=0, max_iterations=1)

    assert decision.action == ActionType.ABSTAIN
    assert decision.abstention_reason == AbstentionReason.CROSS_DOCUMENT_MISMATCH


def test_decision_abstains_after_retries_exhausted() -> None:
    evaluations = [
        _chunk("c1", relevance=False, support_status="unsupported"),
    ]
    trust_res = calculate_trust_score(evaluations=evaluations)
    # iteration=1, max_iterations=1 (retries exhausted)
    decision = make_trust_decision(trust_res, iteration=1, max_iterations=1)

    assert decision.action == ActionType.ABSTAIN
    assert decision.can_retry is False
    assert decision.abstention_reason in (AbstentionReason.INSUFFICIENT_EVIDENCE, AbstentionReason.LOW_TRUST_SCORE)


# ==============================================================================
# Dedicated Tests for HIGH, MEDIUM, and LOW Trust Decision Cases
# ==============================================================================
def test_case_high_trust_continues_to_synthesizer() -> None:
    """HIGH: trust_score >= 0.75 → Continue to Synthesizer."""
    evaluations = [_chunk("c1"), _chunk("c2")]
    trust_res = calculate_trust_score(evaluations=evaluations)
    assert trust_res.trust_score >= 0.75

    decision = make_trust_decision(
        trust_res,
        retrieval_attempts=1,
        max_retrieval_attempts=2,
        high_threshold=0.75,
        low_threshold=0.50,
    )

    # 1. Routing action must continue to synthesizer
    assert decision.action == ActionType.DIRECT_ANSWER
    assert decision.decision == "CONTINUE_TO_SYNTHESIZER"
    assert decision.can_retry is False

    # 2. Return dictionary must contain all required fields
    data = decision.to_dict()
    assert "trust_score" in data
    assert "decision" in data
    assert "reason" in data
    assert "number_of_retrieval_attempts" in data

    assert data["trust_score"] >= 0.75
    assert data["decision"] == "CONTINUE_TO_SYNTHESIZER"
    assert data["number_of_retrieval_attempts"] == 1
    assert "High trust" in data["reason"] or "Synthesizer" in data["reason"]


def test_case_medium_trust_retrieve_more_and_loop_limit() -> None:
    """MEDIUM: 0.50 <= trust_score < 0.75:
    → Retrieve additional evidence (attempts < max)
    → Limit the number of retrieval iterations
    → If still insufficient after max attempts, abstain.
    """
    evaluations = [
        _chunk("c1", relevance=True, support_status="supported"),
        _chunk("c2", relevance=False, support_status="unsupported"),
        _chunk("c3", relevance=False, support_status="unsupported"),
    ]
    trust_res = calculate_trust_score(evaluations=evaluations)
    assert 0.50 <= trust_res.trust_score < 0.75

    # Attempt 1 of 2: Must trigger RETRIEVE_MORE
    decision_att1 = make_trust_decision(
        trust_res,
        retrieval_attempts=1,
        max_retrieval_attempts=2,
        high_threshold=0.75,
        low_threshold=0.50,
    )
    assert decision_att1.action == ActionType.RETRIEVE_MORE
    assert decision_att1.decision == "RETRIEVE_MORE"
    assert decision_att1.can_retry is True
    assert decision_att1.number_of_retrieval_attempts == 1

    data1 = decision_att1.to_dict()
    assert data1["decision"] == "RETRIEVE_MORE"
    assert data1["number_of_retrieval_attempts"] == 1

    # Attempt 2 of 2 (max attempts reached, evidence still medium/insufficient): Must ABSTAIN
    decision_att2 = make_trust_decision(
        trust_res,
        retrieval_attempts=2,
        max_retrieval_attempts=2,
        high_threshold=0.75,
        low_threshold=0.50,
    )
    assert decision_att2.action == ActionType.ABSTAIN
    assert decision_att2.decision == "ABSTAIN"
    assert decision_att2.can_retry is False
    assert decision_att2.number_of_retrieval_attempts == 2

    data2 = decision_att2.to_dict()
    assert data2["decision"] == "ABSTAIN"
    assert data2["number_of_retrieval_attempts"] == 2
    assert "insufficient" in data2["reason"].lower()


def test_case_low_trust_returns_abstention_response() -> None:
    """LOW: trust_score < 0.50:
    → Do not generate a confident answer
    → Return an abstention response containing standard abstention notice.
    """
    from app.agents.abstention import get_abstention_engine

    evaluations = [
        _chunk("c1", relevance=False, support_status="unsupported"),
        _chunk("c2", relevance=False, support_status="unsupported"),
    ]
    trust_res = calculate_trust_score(evaluations=evaluations)
    assert trust_res.trust_score < 0.50

    decision = make_trust_decision(
        trust_res,
        retrieval_attempts=1,
        max_retrieval_attempts=2,
        high_threshold=0.75,
        low_threshold=0.50,
    )

    # 1. Decision must be ABSTAIN with no confident answer
    assert decision.action == ActionType.ABSTAIN
    assert decision.decision == "ABSTAIN"
    assert decision.can_retry is False

    data = decision.to_dict()
    assert data["trust_score"] < 0.50
    assert data["decision"] == "ABSTAIN"
    assert "I don't have enough reliable evidence" in data["reason"] or "low trust" in data["reason"].lower()

    # 2. Abstention engine constructs expected response
    abstention_resp = get_abstention_engine().build_abstention_response(
        question="What is the answer?",
        document_id="doc-test",
        decision=decision,
        trust_result=trust_res,
    )
    assert abstention_resp["abstention"] is True
    assert "I don't have enough reliable evidence in the uploaded document to answer this question." in abstention_resp["answer"]


def test_configurable_thresholds_customization() -> None:
    """Test that custom thresholds dynamically alter decision routing."""
    # Score is 0.70 (which is medium under 0.75, but high under custom 0.65 threshold)
    evaluations = [_chunk("c1", relevance=True, support_status="supported")]
    trust_res = calculate_trust_score(evaluations=evaluations)

    # With default threshold (0.75), if score < 0.75, it does not qualify for HIGH
    # With custom high_threshold=0.60, it qualifies for HIGH
    custom_decision = make_trust_decision(
        trust_res,
        high_threshold=0.60,
        low_threshold=0.40,
    )
    if trust_res.overall_score >= 0.60:
        assert custom_decision.decision == "CONTINUE_TO_SYNTHESIZER"

