"""Unit tests for the Baseline vs Trust-Aware Comparison API and Evaluation API."""

from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from app.api import chat as chat_api
from app.main import app


class FakeRetriever:
    def retrieve(self, question: str, document_id: str, top_k: int = 5):
        return [
            {
                "document_id": document_id,
                "chunk_id": f"{document_id}::c1",
                "filename": "sample.pdf",
                "page_number": 1,
                "source": "sample.pdf#page=1",
                "text": "Self-attention computes representations without recurrence.",
                "score": 0.95,
            }
        ]


class FakeLLM:
    def answer(self, question: str, evidence: list):
        return {
            "answer": "Self-attention computes representations without recurrence.",
            "source_references": [{"source": "sample.pdf#page=1", "filename": "sample.pdf", "page_number": 1}],
            "evidence_references": [{"chunk_id": f"{evidence[0]['document_id']}::c1", "document_id": evidence[0]["document_id"], "source": "sample.pdf#page=1", "page_number": 1}],
        }


def test_compare_endpoint_returns_both_models(monkeypatch) -> None:
    monkeypatch.setattr(chat_api, "get_retriever", lambda: FakeRetriever())
    monkeypatch.setattr(chat_api, "get_groq_client", lambda: FakeLLM())

    client = TestClient(app)
    response = client.post(
        "/api/chat/compare",
        json={"document_id": "test-doc-1", "question": "What does self-attention do?"},
    )

    assert response.status_code == 200
    data = response.json()

    assert "baseline" in data
    assert "trust_aware" in data
    assert "comparison_analysis" in data
    assert data["baseline"]["hallucination_check"] == "NONE (Unverified)"
    assert "safety_impact" in data["comparison_analysis"]
    assert "trust_aware_advantage" in data["comparison_analysis"]


def test_evaluation_benchmark_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/api/evaluation/benchmark")

    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    metrics = data["metrics"]
    assert "hallucination_prevention_rate" in metrics
    assert "abstention_accuracy" in metrics
    assert "contradiction_detection_rate" in metrics
    assert "average_trust_score" in metrics
