"""Tests for Trust-Aware RAG Dashboard API and actual stored system metrics."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.database import query_store
from app.main import app


def test_dashboard_stats_empty_state(tmp_path, monkeypatch) -> None:
    """When no query data exists, dashboard reports has_data=False without inventing numbers."""
    fake_queries_file = tmp_path / "query_history.json"
    fake_metadata_file = tmp_path / "metadata.json"
    fake_uploads_dir = tmp_path / "uploads"
    fake_uploads_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(query_store, "_QUERIES_FILE", str(fake_queries_file))
    monkeypatch.setattr(query_store, "_METADATA_FILE", str(fake_metadata_file))
    monkeypatch.setattr(query_store, "_UPLOADS_DIR", str(fake_uploads_dir))

    client = TestClient(app)
    res = client.get("/api/dashboard/stats")
    assert res.status_code == 200
    data = res.json()
    assert data["has_data"] is False
    assert data["total_questions"] == 0
    assert data["average_trust_score"] is None
    assert data["high_trust_responses"] == 0
    assert data["medium_trust_responses"] == 0
    assert data["low_trust_abstained_responses"] == 0
    assert data["verified_answers"] == 0
    assert data["unsupported_answers"] == 0
    assert data["average_retrieval_relevance"] is None
    assert data["average_response_time_ms"] is None


def test_dashboard_stats_calculates_real_metrics(tmp_path, monkeypatch) -> None:
    """Dashboard computes all 11 metrics accurately from recorded query data."""
    fake_queries_file = tmp_path / "query_history.json"
    fake_metadata_file = tmp_path / "metadata.json"
    fake_uploads_dir = tmp_path / "uploads"
    fake_uploads_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(query_store, "_QUERIES_FILE", str(fake_queries_file))
    monkeypatch.setattr(query_store, "_METADATA_FILE", str(fake_metadata_file))
    monkeypatch.setattr(query_store, "_UPLOADS_DIR", str(fake_uploads_dir))

    # Record 3 distinct query executions
    # 1. High Trust, Verified
    query_store.record_query(
        question="What is the Transformer architecture?",
        document_id="doc-transformer",
        trust_score=0.90,
        trust_level="HIGH",
        verification_status="SUPPORTED",
        is_abstention=False,
        contradictions_count=0,
        average_relevance=0.95,
        response_time_ms=120.0,
        answer="The Transformer uses self-attention.",
    )

    # 2. Medium Trust, Partially Supported
    query_store.record_query(
        question="What is the learning rate schedule?",
        document_id="doc-transformer",
        trust_score=0.65,
        trust_level="MEDIUM",
        verification_status="PARTIALLY_SUPPORTED",
        is_abstention=False,
        contradictions_count=1,
        average_relevance=0.70,
        response_time_ms=250.0,
        answer="Warmup schedule is used.",
    )

    # 3. Low Trust, Abstained, Another Document
    query_store.record_query(
        question="What is the revenue of Apple?",
        document_id="doc-finance",
        trust_score=0.25,
        trust_level="LOW",
        verification_status="ABSTAINED",
        is_abstention=True,
        contradictions_count=0,
        average_relevance=0.20,
        response_time_ms=80.0,
        answer="I don't have enough reliable evidence in the uploaded document to answer this question.",
    )

    client = TestClient(app)
    res = client.get("/api/dashboard/stats")
    assert res.status_code == 200
    data = res.json()

    assert data["has_data"] is True
    # 1. Total documents (2 unique doc IDs from queries)
    assert data["total_documents"] == 2
    # 2. Total questions
    assert data["total_questions"] == 3
    # 3. Average trust score ( (0.90 + 0.65 + 0.25) / 3 = 0.60 )
    assert abs(data["average_trust_score"] - 0.60) < 0.01
    # 4. High-trust responses
    assert data["high_trust_responses"] == 1
    # 5. Medium-trust responses
    assert data["medium_trust_responses"] == 1
    # 6. Low-trust/abstained responses
    assert data["low_trust_abstained_responses"] == 1
    # 7. Verified answers
    assert data["verified_answers"] == 1
    # 8. Unsupported answers
    assert data["unsupported_answers"] == 0
    # 9. Contradictions detected
    assert data["contradictions_detected"] == 1
    # 10. Average retrieval relevance ( (0.95 + 0.70 + 0.20) / 3 = 0.6167 )
    assert abs(data["average_retrieval_relevance"] - 0.6167) < 0.01
    # 11. Average response time ( (120 + 250 + 80) / 3 = 150.0 )
    assert abs(data["average_response_time_ms"] - 150.0) < 1.0


def test_dashboard_filtering_by_document(tmp_path, monkeypatch) -> None:
    """Filtering stats by document_id calculates isolated metrics for that document."""
    fake_queries_file = tmp_path / "query_history.json"
    fake_metadata_file = tmp_path / "metadata.json"
    fake_uploads_dir = tmp_path / "uploads"
    fake_uploads_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(query_store, "_QUERIES_FILE", str(fake_queries_file))
    monkeypatch.setattr(query_store, "_METADATA_FILE", str(fake_metadata_file))
    monkeypatch.setattr(query_store, "_UPLOADS_DIR", str(fake_uploads_dir))

    query_store.record_query(
        question="Q1",
        document_id="doc-a",
        trust_score=0.85,
        trust_level="HIGH",
        verification_status="SUPPORTED",
        is_abstention=False,
        contradictions_count=0,
        average_relevance=0.90,
        response_time_ms=100.0,
    )
    query_store.record_query(
        question="Q2",
        document_id="doc-b",
        trust_score=0.30,
        trust_level="LOW",
        verification_status="ABSTAINED",
        is_abstention=True,
        contradictions_count=1,
        average_relevance=0.20,
        response_time_ms=90.0,
    )

    client = TestClient(app)

    # Filter by doc-a
    res_a = client.get("/api/dashboard/stats?document_id=doc-a")
    assert res_a.status_code == 200
    data_a = res_a.json()
    assert data_a["total_questions"] == 1
    assert data_a["high_trust_responses"] == 1
    assert data_a["average_trust_score"] == 0.85

    # Filter by doc-b
    res_b = client.get("/api/dashboard/stats?document_id=doc-b")
    assert res_b.status_code == 200
    data_b = res_b.json()
    assert data_b["total_questions"] == 1
    assert data_b["low_trust_abstained_responses"] == 1
    assert data_b["average_trust_score"] == 0.30

    # Filter by unknown document
    res_c = client.get("/api/dashboard/stats?document_id=doc-unknown")
    assert res_c.status_code == 200
    data_c = res_c.json()
    assert data_c["has_data"] is False
    assert data_c["total_questions"] == 0


def test_dashboard_documents_endpoint(tmp_path, monkeypatch) -> None:
    """GET /api/dashboard/documents lists all registered documents."""
    fake_queries_file = tmp_path / "query_history.json"
    fake_metadata_file = tmp_path / "metadata.json"
    fake_uploads_dir = tmp_path / "uploads"
    fake_uploads_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(query_store, "_QUERIES_FILE", str(fake_queries_file))
    monkeypatch.setattr(query_store, "_METADATA_FILE", str(fake_metadata_file))
    monkeypatch.setattr(query_store, "_UPLOADS_DIR", str(fake_uploads_dir))

    query_store.record_query(
        question="Test",
        document_id="doc-12345",
        trust_score=0.88,
        trust_level="HIGH",
        verification_status="SUPPORTED",
        is_abstention=False,
        contradictions_count=0,
        average_relevance=0.88,
        response_time_ms=100.0,
    )

    client = TestClient(app)
    res = client.get("/api/dashboard/documents")
    assert res.status_code == 200
    docs = res.json()
    assert any(d["document_id"] == "doc-12345" for d in docs)
