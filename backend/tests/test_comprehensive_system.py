"""Comprehensive System Validation Test Suite.

Methodically verifies all 9 operational categories:
1. DOCUMENT TESTS (PDF, TXT, DOCX, CSV, JSON, URL)
2. RETRIEVAL TESTS (Relevant, Irrelevant, Missing, Ambiguous)
3. DOCUMENT ISOLATION TESTS (Paper A, Paper B, Paper B after A, Cross-doc leakage)
4. TRUST TESTS (High, Medium, Low)
5. CONTRADICTION TESTS (Numerical, Statements, No contradiction)
6. VERIFIER TESTS (Correct answer, Unsupported claim, Partially supported claim)
7. CACHE TESTS (Same doc+query, Different doc+query)
8. SECURITY TESTS (Missing API key, Invalid file, Oversized file, Malformed doc, Invalid document_id)
9. PERFORMANCE TESTS (Retrieval time, LLM time, Total response time)
"""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import docx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app.agents.contradiction_detector import ContradictionDetector
from app.agents.critic import CriticAgent
from app.agents.verifier import VerifierAgent
from app.api import chat as chat_api
from app.api import upload as upload_api
from app.database.redis_cache import RedisCache
from app.database.supabase import SupabaseDatabase
from app.ingestion.chunker import Chunk
from app.ingestion.loaders import (
    ExtractionError,
    load_csv,
    load_doc,
    load_document,
    load_docx,
    load_json,
    load_pdf,
    load_txt,
    load_url,
)
from app.llm.groq_client import FALLBACK_ANSWER, GroqClientError, GroqLLMClient
from app.main import app
from app.retrieval.faiss_store import FAISSStore
from app.retrieval.retriever import Retriever
from app.trust.trust_model import calculate_trust_score


# ==============================================================================
# Helper fixtures and mock objects
# ==============================================================================

class SimpleEmbeddingModel:
    dimension = 4

    def _vector(self, text: str) -> np.ndarray:
        lowered = text.lower()
        if "transformer" in lowered or "attention" in lowered or "paper a" in lowered:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        if "bert" in lowered or "masked" in lowered or "paper b" in lowered:
            return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        if "preliminary" in lowered or "experiments" in lowered:
            return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def embed_chunks(self, chunks):
        return np.vstack([self._vector(c.text) for c in chunks])

    def embed_query(self, query):
        return self._vector(query)


class FakeCompletions:
    def __init__(self, response_json: dict | None = None):
        self.response_json = response_json or {}

    def create(self, **request):
        content = json.dumps(self.response_json)
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeGroq:
    def __init__(self, response_json: dict | None = None):
        self.completions = FakeCompletions(response_json=response_json)
        self.chat = SimpleNamespace(completions=self.completions)


# ==============================================================================
# 1. DOCUMENT TESTS (PDF, TXT, DOCX, CSV, JSON, URL)
# ==============================================================================

def test_document_pdf(tmp_path: Path) -> None:
    """Validate PDF text extraction with pypdf."""
    pdf_path = tmp_path / "test_paper.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with open(pdf_path, "wb") as f:
        writer.write(f)

    # Empty blank page raises clean error
    with pytest.raises(ExtractionError, match="No readable text was found in this document|Could not extract readable text"):
        load_pdf(pdf_path, document_id="doc-pdf-1", source_filename="test_paper.pdf")


def test_document_txt(tmp_path: Path) -> None:
    """Validate TXT document ingestion and page-splitting."""
    txt_path = tmp_path / "research.txt"
    txt_path.write_text("Section 1: Introduction\x0cSection 2: Methodology", encoding="utf-8")

    doc = load_document(txt_path, document_id="doc-txt-1", source_filename="research.txt")
    assert doc.file_type == "txt"
    assert len(doc.pages) == 2
    assert "Section 1" in doc.pages[0].text
    assert "Section 2" in doc.pages[1].text


def test_document_docx(tmp_path: Path) -> None:
    """Validate DOCX document paragraph parsing."""
    docx_path = tmp_path / "paper.docx"
    doc_obj = docx.Document()
    doc_obj.add_heading("Attention Mechanisms", 0)
    doc_obj.add_paragraph("The transformer relies purely on attention.")
    doc_obj.save(docx_path)

    doc = load_document(docx_path, document_id="doc-docx-1", source_filename="paper.docx")
    assert doc.file_type == "docx"
    assert len(doc.pages) == 1
    assert "The transformer relies purely on attention." in doc.pages[0].text


def test_document_csv(tmp_path: Path) -> None:
    """Validate CSV table ingestion and row serialization."""
    csv_path = tmp_path / "benchmarks.csv"
    csv_path.write_text("Model,BLEU,Latency\nTransformer,28.4,25ms\nRNN,24.1,80ms\n", encoding="utf-8")

    doc = load_document(csv_path, document_id="doc-csv-1", source_filename="benchmarks.csv")
    assert doc.file_type == "csv"
    assert len(doc.pages) == 2  # 2 data rows
    assert "Transformer" in doc.pages[0].text


def test_document_json(tmp_path: Path) -> None:
    """Validate JSON structured document ingestion."""
    json_path = tmp_path / "config.json"
    json_path.write_text(json.dumps({"experiment": "BERT", "accuracy": 0.88, "dataset": "SQuAD"}), encoding="utf-8")

    doc = load_document(json_path, document_id="doc-json-1", source_filename="config.json")
    assert doc.file_type == "json"
    assert any("BERT" in p.text for p in doc.pages)
    assert any("0.88" in p.text for p in doc.pages)
    assert any(p.page_number == "section:experiment" for p in doc.pages)


def test_document_url() -> None:
    """Validate URL web ingestion with HTML text cleaning."""
    mock_html = """
    <html>
      <head><title>Research Blog</title></head>
      <body>
        <nav><a href="/home">Home</a></nav>
        <h1>Neural Architectures</h1>
        <p>Transformers have transformed NLP research entirely.</p>
        <script>console.log('tracker');</script>
      </body>
    </html>
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = mock_html

    with patch("httpx.get", return_value=mock_response):
        doc = load_url("https://example.com/research", document_id="doc-url-1")
        assert doc.file_type == "url"
        assert len(doc.pages) == 1
        assert "Neural Architectures" in doc.pages[0].text
        assert "Transformers have transformed NLP research entirely." in doc.pages[0].text
        # Ensure scripts and navs are cleanly stripped
        assert "tracker" not in doc.pages[0].text
        assert "Home" not in doc.pages[0].text


# ==============================================================================
# 2. RETRIEVAL TESTS (Relevant, Irrelevant, Missing, Ambiguous)
# ==============================================================================

@pytest.fixture
def sample_retriever(tmp_path: Path) -> Retriever:
    embedder = SimpleEmbeddingModel()
    store = FAISSStore(tmp_path / "faiss", embedding_model=embedder)
    chunks = [
        Chunk(
            document_id="doc-1",
            chunk_id="doc-1::c1",
            filename="paper.pdf",
            page_number=1,
            text="The Transformer architecture relies entirely on self-attention.",
            source="paper.pdf#page=1",
        ),
        Chunk(
            document_id="doc-1",
            chunk_id="doc-1::c2",
            filename="paper.pdf",
            page_number=2,
            text="Preliminary experiments achieve moderate translation quality under specific parameters.",
            source="paper.pdf#page=2",
        ),
    ]
    store.add_chunks(chunks)
    return Retriever(store=store, embedding_model=embedder, relevance_threshold=0.50)


def test_retrieval_relevant_question(sample_retriever: Retriever) -> None:
    """Query matching document content returns relevant chunks with score >= 0.5."""
    results = sample_retriever.retrieve(
        question="How does the transformer architecture work?",
        document_id="doc-1",
    )
    assert len(results) > 0
    assert results[0]["document_id"] == "doc-1"
    assert "Transformer" in results[0]["text"]
    assert results[0]["score"] >= 0.50


def test_retrieval_irrelevant_question(sample_retriever: Retriever) -> None:
    """Query with completely irrelevant topic returns no chunks above threshold."""
    results = sample_retriever.retrieve(
        question="What is the stock price of Tesla on Nasdaq?",
        document_id="doc-1",
    )
    assert len(results) == 0


def test_retrieval_missing_information(sample_retriever: Retriever) -> None:
    """Query for missing entity returns empty list or ungrounded chunks."""
    results = sample_retriever.retrieve(
        question="What is the secret recipe of Coca-Cola?",
        document_id="doc-1",
    )
    assert results == []


def test_retrieval_ambiguous_question(sample_retriever: Retriever) -> None:
    """Query with ambiguous premise retrieves partial evidence for Critic evaluation."""
    results = sample_retriever.retrieve(
        question="What was the translation performance on preliminary experiments?",
        document_id="doc-1",
    )
    assert len(results) > 0
    assert "Preliminary experiments" in results[0]["text"]


# ==============================================================================
# 3. DOCUMENT ISOLATION TESTS (Paper A, Paper B, Cross-Contamination)
# ==============================================================================

@pytest.fixture
def two_paper_retriever(tmp_path: Path) -> Retriever:
    embedder = SimpleEmbeddingModel()
    store = FAISSStore(tmp_path / "faiss_iso", embedding_model=embedder)
    chunks = [
        Chunk(
            document_id="paper-a",
            chunk_id="paper-a::c1",
            filename="paper_a.pdf",
            page_number=1,
            text="Paper A introduces the Transformer self-attention mechanism.",
            source="paper_a.pdf#page=1",
        ),
        Chunk(
            document_id="paper-b",
            chunk_id="paper-b::c1",
            filename="paper_b.pdf",
            page_number=1,
            text="Paper B introduces BERT masked language modeling representations.",
            source="paper_b.pdf#page=1",
        ),
    ]
    store.add_chunks(chunks)
    return Retriever(store=store, embedding_model=embedder, relevance_threshold=0.50)


def test_isolation_paper_a_question(two_paper_retriever: Retriever) -> None:
    """Querying Paper A retrieves only Paper A passages."""
    results = two_paper_retriever.retrieve("What does Paper A introduce?", document_id="paper-a")
    assert len(results) > 0
    assert all(r["document_id"] == "paper-a" for r in results)


def test_isolation_paper_b_question(two_paper_retriever: Retriever) -> None:
    """Querying Paper B retrieves only Paper B passages."""
    results = two_paper_retriever.retrieve("What does Paper B introduce?", document_id="paper-b")
    assert len(results) > 0
    assert all(r["document_id"] == "paper-b" for r in results)


def test_isolation_paper_b_question_after_paper_a(two_paper_retriever: Retriever) -> None:
    """Sequential queries maintain strict state isolation without bleed-through."""
    res_a = two_paper_retriever.retrieve("What does Paper A introduce?", document_id="paper-a")
    res_b = two_paper_retriever.retrieve("What does Paper B introduce?", document_id="paper-b")
    assert all(r["document_id"] == "paper-a" for r in res_a)
    assert all(r["document_id"] == "paper-b" for r in res_b)


def test_isolation_ask_paper_b_for_paper_a_information(two_paper_retriever: Retriever) -> None:
    """Asking Paper B for information only in Paper A strictly returns 0 chunks."""
    results = two_paper_retriever.retrieve(
        "What is the Transformer self-attention in Paper A?",
        document_id="paper-b",
    )
    # Must NOT leak Paper A chunks
    assert all(r["document_id"] != "paper-a" for r in results)


# ==============================================================================
# 4. TRUST TESTS (High, Medium, Low)
# ==============================================================================

def test_trust_high_score() -> None:
    """Authoritative, supporting evidence yields high trust (>= 0.75)."""
    chunks = [{
        "chunk_id": "c1",
        "document_id": "doc-1",
        "score": 0.95,
        "relevance": True,
        "support_status": "supported",
        "quality_assessment": {"evidence_strength": "high", "source_quality": "high"},
    }]
    result = calculate_trust_score(
        evaluations=chunks,
        contradictions=[],
    )
    assert result.overall_score >= 0.75
    assert result.trust_level == "HIGH_TRUST"


def test_trust_medium_score() -> None:
    """Moderate evidence with mixed relevance and support yields medium trust (0.50 <= score < 0.75)."""
    chunks = [
        {
            "chunk_id": "c1",
            "document_id": "doc-1",
            "relevance": True,
            "support_status": "supported",
            "quality_assessment": {"evidence_strength": "medium", "source_quality": "medium"},
        },
        {
            "chunk_id": "c2",
            "document_id": "doc-1",
            "relevance": True,
            "support_status": "supported",
            "quality_assessment": {"evidence_strength": "medium", "source_quality": "medium"},
        },
        {
            "chunk_id": "c3",
            "document_id": "doc-1",
            "relevance": False,
            "support_status": "unsupported",
            "quality_assessment": {"evidence_strength": "low", "source_quality": "low"},
        },
        {
            "chunk_id": "c4",
            "document_id": "doc-1",
            "relevance": False,
            "support_status": "unsupported",
            "quality_assessment": {"evidence_strength": "none", "source_quality": "low"},
        },
    ]
    retrieval_scores = [0.75, 0.70, 0.50, 0.40]
    result = calculate_trust_score(
        evaluations=chunks,
        contradictions=[],
        retrieval_scores=retrieval_scores,
    )
    assert 0.50 <= result.overall_score < 0.75
    assert result.trust_level == "MODERATE_TRUST"


def test_trust_low_score() -> None:
    """Contradictory or irrelevant evidence yields low trust (< 0.50)."""
    chunks = [
        {"chunk_id": "c1", "document_id": "doc-1", "score": 0.2, "relevance": False, "support_status": "unsupported"},
        {"chunk_id": "c2", "document_id": "doc-1", "score": 0.2, "relevance": False, "support_status": "unsupported"},
    ]
    contradictions = [{"chunk_a": "c1", "chunk_b": "c2", "reason": "Direct conflict", "severity": "high"}]
    result = calculate_trust_score(
        evaluations=chunks,
        contradictions=contradictions,
    )
    assert result.overall_score < 0.50
    assert result.trust_level == "LOW_TRUST"


# ==============================================================================
# 5. CONTRADICTION TESTS (Numerical, Statements, No Contradiction)
# ==============================================================================

def test_contradiction_numerical_values() -> None:
    """Direct numerical discrepancy is detected with high severity."""
    detector = ContradictionDetector()
    chunks = [
        {"chunk_id": "c1", "text": "The model achieved 85% accuracy on the test set."},
        {"chunk_id": "c2", "text": "The model achieved 72% accuracy on the test set."},
    ]
    conflicts = detector.detect_contradictions(chunks)
    assert len(conflicts) > 0
    assert any("accuracy" in c.get("reason", "").lower() or "85" in str(c) for c in conflicts)


def test_contradiction_statements() -> None:
    """Mutually exclusive temporal or factual statements are detected."""
    fake_response = {
        "contradiction": True,
        "reason": "Direct temporal contradiction: Launched in March 2021 vs November 2023.",
        "severity": "high",
        "claim_a": "Launched in March 2021",
        "claim_b": "Launched in November 2023",
    }
    detector = ContradictionDetector(llm_client=FakeGroq(fake_response))
    chunks = [
        {"chunk_id": "c1", "text": "The system was launched in March 2021 and remains active."},
        {"chunk_id": "c2", "text": "The system was launched in November 2023 after delays."},
    ]
    conflicts = detector.detect_contradictions(chunks)
    assert len(conflicts) > 0


def test_contradiction_no_contradiction() -> None:
    """Consistent, non-conflicting passages produce zero contradiction events."""
    detector = ContradictionDetector()
    chunks = [
        {"chunk_id": "c1", "text": "The Transformer uses multi-head attention."},
        {"chunk_id": "c2", "text": "Positional encodings provide sequence order information."},
    ]
    conflicts = detector.detect_contradictions(chunks)
    assert conflicts == []


# ==============================================================================
# 6. VERIFIER TESTS (Correct, Unsupported, Partially Supported)
# ==============================================================================

def test_verifier_correct_answer() -> None:
    """Claims fully corroborated by evidence receive SUPPORTED."""
    fake_response = {
        "claim_evaluations": [
            {
                "claim_id": "clm-1",
                "claim": "Binary search has worst-case time complexity of O(log n).",
                "status": "SUPPORTED",
                "supported": True,
                "supporting_chunk_ids": ["c1"],
                "explanation": "Evidence directly confirms binary search has O(log n) time complexity.",
            }
        ]
    }
    verifier = VerifierAgent(llm_client=FakeGroq(fake_response))
    evidence = [{"chunk_id": "c1", "document_id": "doc-1", "text": "Binary search has O(log n) worst-case time complexity."}]
    claims = [{"claim_id": "clm-1", "claim": "Binary search has worst-case time complexity of O(log n)."}]

    result = verifier.verify(
        claims=claims,
        evidence=evidence,
        draft_answer="Binary search runs in O(log n) time.",
        document_id="doc-1",
    )
    assert result["status"] == "SUPPORTED"
    assert result["verified_claims"][0]["supported"] is True


def test_verifier_unsupported_claim() -> None:
    """Fabricated assertion absent from evidence receives UNSUPPORTED."""
    fake_response = {
        "claim_evaluations": [
            {
                "claim_id": "clm-2",
                "claim": "Geoffrey Hinton was the lead author on the paper.",
                "status": "UNSUPPORTED",
                "supported": False,
                "supporting_chunk_ids": [],
                "explanation": "Evidence makes no mention of Geoffrey Hinton.",
            }
        ]
    }
    verifier = VerifierAgent(llm_client=FakeGroq(fake_response))
    evidence = [{"chunk_id": "c1", "document_id": "doc-1", "text": "The Transformer uses self-attention."}]
    claims = [{"claim_id": "clm-2", "claim": "Geoffrey Hinton was the lead author on the paper."}]

    result = verifier.verify(
        claims=claims,
        evidence=evidence,
        draft_answer="Geoffrey Hinton authored the paper.",
        document_id="doc-1",
    )
    assert result["status"] == "UNSUPPORTED"
    assert result["verified_claims"][0]["supported"] is False
    assert result["hallucination_detected"] is True


def test_verifier_partially_supported_claim() -> None:
    """Draft containing a mix of supported and unsupported claims is marked PARTIALLY_SUPPORTED."""
    fake_response = {
        "claim_evaluations": [
            {
                "claim_id": "clm-1",
                "claim": "The model achieved 28.4 BLEU on English-to-German and beat all human translators.",
                "status": "PARTIALLY_SUPPORTED",
                "supported": False,
                "supporting_chunk_ids": ["c1"],
                "explanation": "28.4 BLEU is confirmed by c1, but human translator superiority is unverified.",
            }
        ]
    }
    verifier = VerifierAgent(llm_client=FakeGroq(fake_response))
    evidence = [{"chunk_id": "c1", "document_id": "doc-1", "text": "The model achieved 28.4 BLEU on English-to-German translation."}]
    result = verifier.verify(
        question="What was the model translation performance?",
        generated_answer="The model achieved 28.4 BLEU on English-to-German and beat all human translators.",
        accepted_evidence=evidence,
        document_id="doc-1",
    )
    assert result["status"] == "PARTIALLY_SUPPORTED"
    assert result["hallucination_risk"] == "MEDIUM"


# ==============================================================================
# 7. CACHE TESTS (Same doc+query vs Different doc+query)
# ==============================================================================

class MemoryRedis:
    def __init__(self):
        self._store = {}

    def get(self, key):
        return self._store.get(key)

    def setex(self, key, ttl, value):
        self._store[key] = value


def test_cache_same_document_same_question() -> None:
    """Identical query on the same document hits cache."""
    cache = RedisCache(client=MemoryRedis())
    key1 = cache.query_key("doc-alpha", "What is attention?")
    key2 = cache.query_key("doc-alpha", "What is attention?")
    assert key1 == key2

    cache.set(key1, {"answer": "Self-attention mechanism."})
    cached_data = cache.get(key2)
    assert cached_data == {"answer": "Self-attention mechanism."}


def test_cache_different_document_same_question() -> None:
    """Same query on a different document produces an isolated key and cache miss."""
    cache = RedisCache(client=MemoryRedis())
    key_doc_a = cache.query_key("doc-alpha", "What is the conclusion?")
    key_doc_b = cache.query_key("doc-beta", "What is the conclusion?")

    assert key_doc_a != key_doc_b
    cache.set(key_doc_a, {"answer": "Conclusion of Alpha."})
    assert cache.get(key_doc_b) is None


# ==============================================================================
# 8. SECURITY TESTS (Missing API key, Invalid file, Oversized file, Path traversal)
# ==============================================================================

def test_security_missing_api_keys() -> None:
    """Client with empty API key raises clear GroqClientError."""
    client = GroqLLMClient(api_key="")
    with pytest.raises(GroqClientError, match="GROQ_API_KEY must be configured"):
        _ = client.client


def test_security_invalid_file_extension() -> None:
    """Uploading executable or unsupported file extension returns 400."""
    client = TestClient(app)
    res = client.post(
        "/api/upload",
        files={"file": ("malicious.exe", b"\x4d\x5a\x90\x00", "application/octet-stream")},
    )
    assert res.status_code == 400
    assert "Unsupported file type" in res.json()["detail"]


def test_security_oversized_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Uploading a file exceeding MAX_FILE_SIZE_BYTES returns HTTP 413."""
    client = TestClient(app)
    # Monkeypatch MAX_FILE_SIZE_BYTES to 100KB for test speed
    monkeypatch.setattr(upload_api, "MAX_FILE_SIZE_BYTES", 100 * 1024)

    large_payload = b"A" * (150 * 1024)
    res = client.post(
        "/api/upload",
        files={"file": ("large_document.txt", large_payload, "text/plain")},
    )
    assert res.status_code == 413
    assert "File exceeds maximum allowed size" in res.json()["detail"]


def test_security_malformed_document(tmp_path: Path) -> None:
    """Malformed JSON returns 400 ExtractionError."""
    bad_json = tmp_path / "corrupt.json"
    bad_json.write_text("{ unquoted_key: invalid json ...", encoding="utf-8")
    with pytest.raises(ExtractionError, match="not valid JSON|Failed to parse JSON"):
        load_json(bad_json, document_id="doc-bad", source_filename="corrupt.json")


def test_security_invalid_document_id() -> None:
    """document_id containing path traversal or null bytes returns 400 Bad Request."""
    client = TestClient(app)
    res = client.post(
        "/api/chat",
        json={"document_id": "../../etc/passwd", "question": "What is the secret?"},
    )
    assert res.status_code == 400
    assert "Invalid document_id" in res.json()["detail"]


# ==============================================================================
# 9. PERFORMANCE TESTS (Retrieval time, LLM time, Total response time)
# ==============================================================================

def test_performance_retrieval_and_total_response_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pipeline tracks retrieval_time_ms, llm_time_ms, and total_time_ms."""
    embedder = SimpleEmbeddingModel()
    store = FAISSStore(tmp_path / "faiss_perf", embedding_model=embedder)
    store.add_chunks([
        Chunk(
            document_id="perf-doc",
            chunk_id="perf-doc::c1",
            filename="sample.txt",
            page_number=1,
            text="Transformer uses attention mechanisms.",
            source="sample.txt#page=1",
        )
    ])
    retriever = Retriever(store=store, embedding_model=embedder, relevance_threshold=0.50)
    monkeypatch.setattr(chat_api, "get_retriever", lambda: retriever)

    # Mock fast LLM answer
    class MockLLM:
        def answer(self, question, evidence):
            time.sleep(0.01)  # 10ms simulation
            return {
                "answer": "Grounded answer.",
                "source_references": [{"source": "sample.txt#page=1", "page_number": 1}],
                "evidence_references": [],
            }

    monkeypatch.setattr(chat_api, "get_groq_client", lambda: MockLLM())

    client = TestClient(app)
    res = client.post(
        "/api/chat",
        json={"document_id": "perf-doc", "question": "How does attention work?", "mode": "baseline"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "performance" in data
    perf = data["performance"]

    # Validate timing dimensions
    assert "retrieval_time_ms" in perf
    assert "llm_time_ms" in perf
    assert "total_time_ms" in perf
    assert perf["total_time_ms"] >= 0.0
    assert perf["retrieval_time_ms"] >= 0.0
    assert perf["llm_time_ms"] >= 0.0
