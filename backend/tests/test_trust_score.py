"""Unit tests for Trust Score Model and Explanation."""

from __future__ import annotations

import pytest

from app.trust.trust_model import (
    HIGH_TRUST_THRESHOLD,
    MODERATE_TRUST_THRESHOLD,
    calculate_trust_score,
    explain_trust_score,
)


def _sample_chunk(
    chunk_id: str = "chunk-1",
    document_id: str = "doc-1",
    relevance: bool = True,
    support_status: str = "supported",
    evidence_strength: str = "high",
    source_quality: str = "high",
    potential_outdated: bool = False,
    contradiction_status: str = "none",
) -> dict:
    return {
        "document_id": document_id,
        "chunk_id": chunk_id,
        "relevance": relevance,
        "support_status": support_status,
        "quality_assessment": {
            "evidence_strength": evidence_strength,
            "source_quality": source_quality,
            "potential_outdated": potential_outdated,
            "details": "Validated test chunk.",
        },
        "contradiction_status": contradiction_status,
        "explanation": "Test explanation.",
    }


def test_trust_score_high_for_authoritative_relevant_evidence() -> None:
    evaluations = [
        _sample_chunk("c1", "doc-1", relevance=True, evidence_strength="high", source_quality="high"),
        _sample_chunk("c2", "doc-1", relevance=True, evidence_strength="high", source_quality="high"),
    ]
    result = calculate_trust_score(evaluations=evaluations, contradictions=[])

    assert result.overall_score >= HIGH_TRUST_THRESHOLD
    assert result.trust_level == "HIGH_TRUST"
    assert result.components.relevance_score == 1.0
    assert result.components.source_quality_score == 1.0
    assert result.components.consistency_score == 1.0
    assert result.components.freshness_score == 1.0
    assert result.factors.relevant_chunks == 2
    assert result.factors.contradicting_chunks == 0
    assert len(result.warnings) == 0


def test_trust_score_low_when_all_chunks_irrelevant() -> None:
    evaluations = [
        _sample_chunk("c1", "doc-1", relevance=False, support_status="unsupported"),
        _sample_chunk("c2", "doc-1", relevance=False, support_status="unsupported"),
    ]
    result = calculate_trust_score(evaluations=evaluations, contradictions=[])

    assert result.overall_score < MODERATE_TRUST_THRESHOLD
    assert result.trust_level == "LOW_TRUST"
    assert result.components.relevance_score == 0.0
    assert result.factors.relevant_chunks == 0
    assert "No relevant chunks found" in result.warnings[0]


def test_trust_score_penalized_for_contradictions() -> None:
    evaluations = [
        _sample_chunk("c1", "doc-1", relevance=True, contradiction_status="contradiction_detected: conflicting dates"),
        _sample_chunk("c2", "doc-1", relevance=True, contradiction_status="contradiction_detected: conflicting dates"),
    ]
    contradictions = [
        {"chunk_id_a": "c1", "chunk_id_b": "c2", "explanation": "Conflicting dates reported"}
    ]
    result = calculate_trust_score(evaluations=evaluations, contradictions=contradictions)

    # Consistency score must drop sharply
    assert result.components.consistency_score <= 0.55
    assert result.trust_level != "HIGH_TRUST"
    assert result.factors.contradicting_chunks >= 1
    assert any("contradiction detected" in w.lower() for w in result.warnings)


def test_trust_score_penalized_for_outdated_chunks() -> None:
    evaluations = [
        _sample_chunk("c1", "doc-1", relevance=True, potential_outdated=True),
        _sample_chunk("c2", "doc-1", relevance=True, potential_outdated=True),
    ]
    result = calculate_trust_score(evaluations=evaluations, contradictions=[])

    assert result.components.freshness_score < 1.0
    assert result.factors.outdated_chunks == 2
    assert any("outdated" in w.lower() for w in result.warnings)


def test_trust_score_strictly_rejects_unrelated_documents() -> None:
    evaluations = [
        {
            "document_id": "doc-other",
            "chunk_id": "c-cross",
            "relevance": False,
            "support_status": "unrelated_document",
            "quality_assessment": {"evidence_strength": "none", "source_quality": "low", "potential_outdated": False},
            "contradiction_status": "none",
            "explanation": "Rejected: chunk belongs to unrelated document.",
        }
    ]
    result = calculate_trust_score(evaluations=evaluations)

    assert result.overall_score == 0.0
    assert result.trust_level == "LOW_TRUST"
    assert result.factors.unrelated_chunks == 1
    assert any("unrelated" in w.lower() for w in result.warnings)


def test_explain_trust_score_provides_structured_details() -> None:
    evaluations = [
        _sample_chunk("c1", "doc-1", relevance=True, evidence_strength="high", source_quality="medium"),
    ]
    result = calculate_trust_score(evaluations=evaluations)
    explanation = explain_trust_score(result)

    assert "overall_score" in explanation
    assert "trust_percentage" in explanation
    assert "trust_level" in explanation
    assert "components" in explanation
    assert "factors" in explanation
    assert "rationale" in explanation
    assert isinstance(explanation["warnings"], list)


# ==============================================================================
# High, Medium, and Low Trust Scenario Tests & Schema Verification
# ==============================================================================
def test_high_trust_scenario() -> None:
    """Test HIGH_TRUST scenario: authoritative, highly relevant evidence with strong support and zero contradictions."""
    evaluations = [
        _sample_chunk("c1", "doc-1", relevance=True, support_status="supported", evidence_strength="high", source_quality="high"),
        _sample_chunk("c2", "doc-1", relevance=True, support_status="supported", evidence_strength="high", source_quality="high"),
        _sample_chunk("c3", "doc-1", relevance=True, support_status="supported", evidence_strength="high", source_quality="high"),
    ]
    retrieval_scores = [0.95, 0.92, 0.90]

    result = calculate_trust_score(
        evaluations=evaluations,
        contradictions=[],
        retrieval_scores=retrieval_scores,
    )

    # 1. Produced trust score must be between 0 and 1
    assert 0.0 <= result.trust_score <= 1.0
    assert result.trust_score >= HIGH_TRUST_THRESHOLD  # >= 0.75
    assert result.trust_level == "HIGH_TRUST"

    # 2. Output dictionary must contain required user fields
    data = result.to_dict()
    assert "trust_score" in data
    assert "relevance" in data
    assert "source_quality" in data
    assert "contradiction_ratio" in data
    assert "evidence_agreement" in data
    assert "scoring_method" in data
    assert "explanation" in data

    # 3. Individual feature values must be in [0.0, 1.0]
    assert 0.0 <= data["relevance"] <= 1.0
    assert 0.0 <= data["source_quality"] <= 1.0
    assert data["contradiction_ratio"] == 0.0
    assert data["evidence_agreement"] >= 0.85
    assert data["scoring_method"] in ("weighted_fallback", "ml_xgboost")
    assert len(result.warnings) == 0


def test_medium_trust_scenario() -> None:
    """Test MODERATE_TRUST scenario: mixed evidence with partial relevance and support."""
    # 2 relevant chunks, 2 unsupported chunks -> moderate relevance (0.50)
    evaluations = [
        _sample_chunk("c1", "doc-1", relevance=True, support_status="supported", evidence_strength="medium", source_quality="medium"),
        _sample_chunk("c2", "doc-1", relevance=True, support_status="supported", evidence_strength="medium", source_quality="medium"),
        _sample_chunk("c3", "doc-1", relevance=False, support_status="unsupported", evidence_strength="low", source_quality="low"),
        _sample_chunk("c4", "doc-1", relevance=False, support_status="unsupported", evidence_strength="none", source_quality="low"),
    ]
    retrieval_scores = [0.75, 0.70, 0.50, 0.40]

    result = calculate_trust_score(
        evaluations=evaluations,
        contradictions=[],
        retrieval_scores=retrieval_scores,
    )

    assert 0.0 <= result.trust_score <= 1.0
    assert MODERATE_TRUST_THRESHOLD <= result.trust_score < HIGH_TRUST_THRESHOLD  # 0.50 <= score < 0.75
    assert result.trust_level == "MODERATE_TRUST"

    data = result.to_dict()
    assert 0.0 <= data["relevance"] <= 1.0
    assert data["contradiction_ratio"] == 0.0
    assert "MODERATE_TRUST" in data["explanation"]


def test_low_trust_scenario_due_to_irrelevance() -> None:
    """Test LOW_TRUST scenario: completely irrelevant retrieved evidence."""
    evaluations = [
        _sample_chunk("c1", "doc-1", relevance=False, support_status="unsupported", evidence_strength="none", source_quality="low"),
        _sample_chunk("c2", "doc-1", relevance=False, support_status="unsupported", evidence_strength="none", source_quality="low"),
    ]
    result = calculate_trust_score(evaluations=evaluations, contradictions=[])

    assert 0.0 <= result.trust_score <= 1.0
    assert result.trust_score < MODERATE_TRUST_THRESHOLD  # < 0.50
    assert result.trust_level == "LOW_TRUST"
    assert result.features.relevance == 0.0
    assert result.features.evidence_support == 0.0


def test_low_trust_scenario_due_to_contradictions() -> None:
    """Test LOW_TRUST scenario: evidence contains explicit contradictory claims."""
    evaluations = [
        _sample_chunk("c1", "doc-1", relevance=True, support_status="contradicted", contradiction_status="contradiction_detected: 85% vs 72%"),
        _sample_chunk("c2", "doc-1", relevance=True, support_status="contradicted", contradiction_status="contradiction_detected: 85% vs 72%"),
    ]
    contradictions = [
        {
            "contradiction": True,
            "chunk_a": "c1",
            "chunk_b": "c2",
            "reason": "Direct numerical contradiction on accuracy: 85% vs 72%",
            "severity": "high",
        }
    ]

    result = calculate_trust_score(evaluations=evaluations, contradictions=contradictions)

    assert 0.0 <= result.trust_score <= 1.0
    assert result.trust_score < MODERATE_TRUST_THRESHOLD  # Severely penalized by contradiction
    assert result.trust_level == "LOW_TRUST"
    assert result.features.contradiction_ratio > 0.0
    assert any("contradiction" in w.lower() for w in result.warnings)


def test_trust_score_with_verifier_agreement() -> None:
    """Test that verifier agreement is incorporated into features when available."""
    evaluations = [
        _sample_chunk("c1", "doc-1", relevance=True, support_status="supported", evidence_strength="high", source_quality="high"),
    ]
    result = calculate_trust_score(evaluations=evaluations, verifier_score=0.92)

    assert result.features.verifier_agreement == 0.92
    data = result.to_dict()
    assert data["verifier_agreement"] == 0.92


def test_configurable_thresholds_labeling() -> None:
    """Test that operational thresholds (0.75 and 0.50) are explicitly documented as configurable heuristics."""
    evaluations = [_sample_chunk("c1", "doc-1", relevance=True)]
    result = calculate_trust_score(evaluations=evaluations)
    data = result.to_dict()

    # Threshold configuration must be present and labeled as uncalibrated heuristic
    assert "threshold_config" in data
    cfg = data["threshold_config"]
    assert cfg["high_threshold"] == 0.75
    assert cfg["low_threshold"] == 0.50
    assert "uncalibrated" in cfg["calibration_status"].lower() or "configurable" in cfg["calibration_status"].lower()

    # Verify custom thresholds can be passed
    custom_res = calculate_trust_score(evaluations=evaluations, high_threshold=0.85, low_threshold=0.60)
    assert custom_res.high_threshold == 0.85
    assert custom_res.low_threshold == 0.60


def test_xgboost_fallback_separation() -> None:
    """Verify that ML prediction and fallback scoring are kept separated and handle missing model cleanly."""
    from app.trust.trust_model import (
        TrustFeatures,
        calculate_fallback_trust_score,
        predict_xgboost_trust_score,
    )

    features = TrustFeatures(
        relevance=0.85,
        evidence_support=0.80,
        source_quality=0.90,
        evidence_agreement=0.85,
        contradiction_ratio=0.0,
        retrieval_confidence=0.80,
        verifier_agreement=0.90,
    )

    # Missing model must return None without raising an unhandled exception
    ml_result = predict_xgboost_trust_score(features, model_path="/nonexistent/model.json")
    assert ml_result is None

    # Fallback formula must calculate an objective score in [0.0, 1.0]
    fallback_score = calculate_fallback_trust_score(features)
    assert 0.0 <= fallback_score <= 1.0
    assert fallback_score >= 0.75

