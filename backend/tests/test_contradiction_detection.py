"""Comprehensive test suite for Explicit Contradiction Detection in Trust-Aware RAG.

Tests covers:
1. No contradiction (complementary facts about the same entity)
2. Direct numerical contradiction (e.g. 85% vs 72% accuracy)
3. Contradictory statements (explicit conflicting claims / polar negations / semantic)
4. Unrelated statements (completely different topics, zero LLM calls)
5. Same fact expressed differently (synonyms, formatting, "85 percent" vs "85%")
6. Different metrics not contradictory (training accuracy vs test accuracy)
7. Critic Agent integration and persistence
8. Contradiction API endpoints
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
import uuid

import pytest
from fastapi.testclient import TestClient

from app.agents.contradiction_detector import (
    ContradictionDetector,
    ContradictionResult,
    check_deterministic_contradiction,
    get_contradiction_detector,
)
from app.agents.critic import CriticAgent
from app.database.metadata import get_contradictions, store_contradictions
from app.main import app


class FakeCompletions:
    def __init__(self, response_json: dict[str, Any]):
        self.response_json = response_json
        self.call_count = 0

    def create(self, **request):
        self.call_count += 1
        content = json.dumps(self.response_json)
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeGroq:
    def __init__(self, response_json: dict[str, Any]):
        self.completions = FakeCompletions(response_json)
        self.chat = SimpleNamespace(completions=self.completions)

    @property
    def call_count(self) -> int:
        return self.completions.call_count


# ==============================================================================
# 1. No Contradiction Test
# ==============================================================================
def test_no_contradiction() -> None:
    """Test that complementary, non-conflicting facts about the same entity are not contradictory."""
    chunk_a = {
        "chunk_id": "chunk-arch",
        "text": "The Transformer architecture relies entirely on multi-head self-attention mechanisms without recurrence.",
    }
    chunk_b = {
        "chunk_id": "chunk-hardware",
        "text": "The Transformer was trained on 8 NVIDIA P100 GPUs for 3.5 days using the Adam optimizer.",
    }

    mock_response = {
        "contradiction": False,
        "reason": "The statements provide complementary information about architecture and training hardware.",
        "severity": "none",
        "claim_a": "",
        "claim_b": "",
    }
    fake_llm = FakeGroq(mock_response)
    detector = ContradictionDetector(llm_client=fake_llm)

    result = detector.compare_pair(
        chunk_a_id=chunk_a["chunk_id"],
        chunk_a_text=chunk_a["text"],
        chunk_b_id=chunk_b["chunk_id"],
        chunk_b_text=chunk_b["text"],
    )

    assert result.contradiction is False
    assert result.severity == "none"
    assert "complementary" in result.reason.lower() or "no contradiction" in result.reason.lower()

    # detect_contradictions should return empty list
    contradictions = detector.detect_contradictions([chunk_a, chunk_b])
    assert len(contradictions) == 0


# ==============================================================================
# 2. Direct Numerical Contradiction Test
# ==============================================================================
def test_direct_numerical_contradiction() -> None:
    """Test direct numerical contradiction (e.g. Document A: '85%' vs Document B: '72%').

    Must be detected deterministically without requiring an LLM call.
    """
    chunk_a_id = "doc_a"
    chunk_a_text = "The accuracy is 85%."

    chunk_b_id = "doc_b"
    chunk_b_text = "The accuracy is 72%."

    # Fake LLM with 0 allowed calls to verify deterministic execution
    fake_llm = FakeGroq({"contradiction": False})
    detector = ContradictionDetector(llm_client=fake_llm)

    result = detector.compare_pair(
        chunk_a_id=chunk_a_id,
        chunk_a_text=chunk_a_text,
        chunk_b_id=chunk_b_id,
        chunk_b_text=chunk_b_text,
    )

    # Verify no LLM call was needed
    assert fake_llm.call_count == 0

    # Verify result fields match prompt requirements
    assert result.contradiction is True
    assert result.chunk_a == "doc_a"
    assert result.chunk_b == "doc_b"
    assert result.severity == "high"
    assert "accuracy" in result.reason.lower()
    assert "85%" in result.reason
    assert "72%" in result.reason

    # Verify dictionary serialization contains expected keys
    data = result.to_dict()
    assert data["contradiction"] is True
    assert data["chunk_a"] == "doc_a"
    assert data["chunk_b"] == "doc_b"
    assert "reason" in data
    assert data["severity"] == "high"
    # Backwards compatibility aliases
    assert data["chunk_id_a"] == "doc_a"
    assert data["chunk_id_b"] == "doc_b"
    assert data["explanation"] == data["reason"]

    # Verify batch pairwise detection
    chunks = [
        {"chunk_id": chunk_a_id, "text": chunk_a_text},
        {"chunk_id": chunk_b_id, "text": chunk_b_text},
    ]
    batch_res = detector.detect_contradictions(chunks)
    assert len(batch_res) == 1
    assert batch_res[0]["contradiction"] is True


def test_direct_numerical_contradiction_reverse_format() -> None:
    """Test numerical contradiction where metric comes after value (e.g. '85% accuracy' vs '72% accuracy')."""
    detector = ContradictionDetector()
    result = detector.compare_pair(
        chunk_a_id="c1",
        chunk_a_text="The proposed model achieved 85% accuracy on ImageNet.",
        chunk_b_id="c2",
        chunk_b_text="The proposed model achieved 72% accuracy on ImageNet.",
    )
    assert result.contradiction is True
    assert result.severity == "high"
    assert "accuracy" in result.reason.lower()


# ==============================================================================
# 3. Contradictory Statements Test
# ==============================================================================
def test_contradictory_statements() -> None:
    """Test contradictory factual statements via polar negations and semantic analysis."""
    # Sub-case A: Polar negation (approved vs rejected) - Deterministic
    fake_llm = FakeGroq({"contradiction": False})
    detector = ContradictionDetector(llm_client=fake_llm)

    result_polar = detector.compare_pair(
        chunk_a_id="chunk-board-1",
        chunk_a_text="The board of directors approved the merger agreement during the meeting.",
        chunk_b_id="chunk-board-2",
        chunk_b_text="The board of directors rejected the merger agreement during the meeting.",
    )
    assert result_polar.contradiction is True
    assert result_polar.severity == "high"
    assert fake_llm.call_count == 0  # Deterministic negation detection

    # Sub-case B: Enabled vs Disabled
    result_enabled = detector.compare_pair(
        chunk_a_id="cfg-1",
        chunk_a_text="Asynchronous replication is enabled by default in the cluster settings.",
        chunk_b_id="cfg-2",
        chunk_b_text="Asynchronous replication is disabled by default in the cluster settings.",
    )
    assert result_enabled.contradiction is True
    assert result_enabled.severity == "high"

    # Sub-case C: Semantic contradiction via Groq/Llama
    semantic_response = {
        "contradiction": True,
        "reason": "Statement A states headquarters is in Tokyo, while Statement B states it is in Berlin.",
        "severity": "high",
        "claim_a": "Headquarters in Tokyo, Japan",
        "claim_b": "Headquarters in Berlin, Germany",
    }
    fake_llm_semantic = FakeGroq(semantic_response)
    detector_semantic = ContradictionDetector(llm_client=fake_llm_semantic)

    result_semantic = detector_semantic.compare_pair(
        chunk_a_id="hq-1",
        chunk_a_text="The company global headquarters is situated in Tokyo, Japan.",
        chunk_b_id="hq-2",
        chunk_b_text="The company global headquarters is situated in Berlin, Germany.",
    )
    assert result_semantic.contradiction is True
    assert result_semantic.severity == "high"
    assert "tokyo" in result_semantic.reason.lower() or "berlin" in result_semantic.reason.lower()
    assert fake_llm_semantic.call_count == 1


# ==============================================================================
# 4. Unrelated Statements Test
# ==============================================================================
def test_unrelated_statements() -> None:
    """Test that completely unrelated statements are never flagged as contradictory."""
    chunk_a = "To bake artisanal sourdough bread, combine organic flour with water, salt, and active starter."
    chunk_b = "Transformers utilize multi-head scaled dot-product attention across hidden representation dimensions."

    fake_llm = FakeGroq({"contradiction": True})  # Should NOT be called
    detector = ContradictionDetector(llm_client=fake_llm)

    result = detector.compare_pair(
        chunk_a_id="bread",
        chunk_a_text=chunk_a,
        chunk_b_id="transformers",
        chunk_b_text=chunk_b,
    )

    # Must be resolved deterministically without calling LLM
    assert fake_llm.call_count == 0
    assert result.contradiction is False
    assert result.severity == "none"
    assert "unrelated" in result.reason.lower()


# ==============================================================================
# 5. Same Fact Expressed Differently Test
# ==============================================================================
def test_same_fact_expressed_differently() -> None:
    """Test that identical factual claims expressed with different words or formatting are not contradictory."""
    # Sub-case A: Numerical equivalence ('85%' vs '85 percent')
    fake_llm = FakeGroq({"contradiction": True})  # Should NOT be called
    detector = ContradictionDetector(llm_client=fake_llm)

    result_num = detector.compare_pair(
        chunk_a_id="stat-1",
        chunk_a_text="The reported accuracy was 85%.",
        chunk_b_id="stat-2",
        chunk_b_text="The measured accuracy was 85 percent.",
    )
    assert fake_llm.call_count == 0
    assert result_num.contradiction is False
    assert result_num.severity == "none"
    assert "agree" in result_num.reason.lower() or "consistent" in result_num.reason.lower()

    # Sub-case B: Paraphrased entities with same core meaning
    result_paraphrase = detector.compare_pair(
        chunk_a_id="geo-1",
        chunk_a_text="Paris is the capital city of France.",
        chunk_b_id="geo-2",
        chunk_b_text="The capital city of France is Paris.",
    )
    assert result_paraphrase.contradiction is False
    assert result_paraphrase.severity == "none"


# ==============================================================================
# 6. Different Information / Subsets Not Contradictory Test
# ==============================================================================
def test_different_metrics_not_contradictory() -> None:
    """Test that distinct metric modifiers (e.g. 'training accuracy' vs 'test accuracy') are not flagged."""
    chunk_a = "The model reached a training accuracy of 95% after 20 epochs."
    chunk_b = "The model achieved a test accuracy of 82% on the unseen evaluation split."

    # Mock LLM indicates complementary evaluation metrics
    fake_llm = FakeGroq({
        "contradiction": False,
        "reason": "Training accuracy and test accuracy are distinct metrics; no conflict.",
        "severity": "none",
    })
    detector = ContradictionDetector(llm_client=fake_llm)

    result = detector.compare_pair(
        chunk_a_id="train-acc",
        chunk_a_text=chunk_a,
        chunk_b_id="test-acc",
        chunk_b_text=chunk_b,
    )
    assert result.contradiction is False
    assert result.severity == "none"


# ==============================================================================
# 7. Critic Agent Integration & Persistence Test
# ==============================================================================
def test_critic_agent_evaluates_and_stores_contradiction() -> None:
    """Verify CriticAgent detects contradictions, marks chunk statuses, and stores them in metadata."""
    doc_id = f"doc-test-{uuid.uuid4().hex[:8]}"

    evidence = [
        {
            "document_id": doc_id,
            "chunk_id": "chunk-85",
            "filename": "report.pdf",
            "text": "The model accuracy is 85%.",
        },
        {
            "document_id": doc_id,
            "chunk_id": "chunk-72",
            "filename": "report.pdf",
            "text": "The model accuracy is 72%.",
        },
    ]

    critic_mock_evals = {
        "evaluations": [
            {
                "chunk_id": "chunk-85",
                "relevance": True,
                "support_status": "supported",
                "quality_assessment": {"evidence_strength": "high", "source_quality": "high"},
                "contradiction_status": "none",
                "explanation": "Reports 85% accuracy.",
            },
            {
                "chunk_id": "chunk-72",
                "relevance": True,
                "support_status": "supported",
                "quality_assessment": {"evidence_strength": "high", "source_quality": "high"},
                "contradiction_status": "none",
                "explanation": "Reports 72% accuracy.",
            },
        ],
        "contradictions": [],
    }

    critic = CriticAgent(llm_client=FakeGroq(critic_mock_evals))
    result = critic.evaluate(
        question="What is the accuracy?",
        evidence=evidence,
        document_id=doc_id,
    )

    # 1. Verify contradictions list is populated
    assert len(result["contradictions"]) == 1
    contra = result["contradictions"][0]
    assert contra["contradiction"] is True
    assert "85%" in contra["reason"] and "72%" in contra["reason"]
    assert contra["chunk_a"] == "chunk-85"
    assert contra["chunk_b"] == "chunk-72"

    # 2. Verify evaluations marked conflicting chunks as contradicted
    evals = {e["chunk_id"]: e for e in result["evaluations"]}
    assert evals["chunk-85"]["support_status"] == "contradicted"
    assert "contradiction_detected" in evals["chunk-85"]["contradiction_status"]
    assert evals["chunk-72"]["support_status"] == "contradicted"
    assert "contradiction_detected" in evals["chunk-72"]["contradiction_status"]

    # 3. Verify summary metrics
    assert result["summary"]["has_contradictions"] is True
    assert result["summary"]["contradicting_chunks"] == 2

    # 4. Verify persistent storage in metadata.py
    stored = get_contradictions(doc_id)
    assert len(stored) >= 1
    assert any(c.get("chunk_a") == "chunk-85" and c.get("chunk_b") == "chunk-72" for c in stored)


# ==============================================================================
# 8. Contradiction API Endpoints Test
# ==============================================================================
def test_contradiction_api_endpoints() -> None:
    """Test /api/critic/contradictions/detect and /api/critic/contradictions/{doc_id} routes."""
    client = TestClient(app)

    # Test direct detection endpoint
    detect_resp = client.post(
        "/api/critic/contradictions/detect",
        json={
            "chunk_a": "The system accuracy is 85%.",
            "chunk_b": "The system accuracy is 72%.",
            "chunk_a_id": "chunk-a",
            "chunk_b_id": "chunk-b",
        },
    )
    assert detect_resp.status_code == 200
    data = detect_resp.json()
    assert data["contradiction"] is True
    assert data["severity"] == "high"
    assert data["chunk_a"] == "chunk-a"
    assert data["chunk_b"] == "chunk-b"
    assert "85%" in data["reason"] and "72%" in data["reason"]

    # Test retrieval endpoint for document
    test_doc = f"api-doc-{uuid.uuid4().hex[:6]}"
    store_contradictions(
        test_doc,
        [{"contradiction": True, "chunk_a": "c1", "chunk_b": "c2", "reason": "Test conflict", "severity": "high"}],
    )
    get_resp = client.get(f"/api/critic/contradictions/{test_doc}")
    assert get_resp.status_code == 200
    get_data = get_resp.json()
    assert get_data["document_id"] == test_doc
    assert len(get_data["contradictions"]) >= 1
    assert get_data["contradictions"][0]["reason"] == "Test conflict"
