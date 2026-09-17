"""End-to-end API tests for the baseline RAG pipeline."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.api import chat as chat_api
from app.ingestion.chunker import chunk_document
from app.ingestion.loaders import load_document
from app.main import app


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def retrieve(self, question, document_id):
        self.calls.append((question, document_id))
        return self.results


class FakeLLM:
    def __init__(self):
        self.evidence = None

    def answer(self, question, evidence):
        self.evidence = evidence
        return {
            "answer": "The Transformer is based on attention mechanisms.",
            "source_references": [
                {
                    "source": evidence[0]["source"],
                    "filename": evidence[0]["filename"],
                    "page_number": evidence[0]["page_number"],
                }
            ],
            "evidence_references": [
                {
                    "chunk_id": evidence[0]["chunk_id"],
                    "document_id": evidence[0]["document_id"],
                    "source": evidence[0]["source"],
                    "page_number": evidence[0]["page_number"],
                }
            ],
        }


def _paper_evidence(document_id="attention-paper"):
    return [
        {
            "document_id": document_id,
            "chunk_id": f"{document_id}::chunk-0000",
            "filename": "attention_is_all_you_need.pdf",
            "page_number": 1,
            "text": "The Transformer is based solely on attention mechanisms.",
            "source": "attention_is_all_you_need.pdf#page=1",
            "score": 0.98,
        }
    ]


def test_chat_runs_retrieval_then_llm_with_research_paper_evidence(monkeypatch) -> None:
    retriever = FakeRetriever(_paper_evidence())
    llm = FakeLLM()
    monkeypatch.setattr(chat_api, "get_retriever", lambda: retriever)
    monkeypatch.setattr(chat_api, "get_groq_client", lambda: llm)

    response = TestClient(app).post(
        "/api/chat",
        json={
            "document_id": "attention-paper",
            "question": "What is the Transformer based solely on?",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("The Transformer")
    assert body["document_id"] == "attention-paper"
    assert body["sources"][0]["page_number"] == 1
    assert body["evidence"][0]["chunk_id"] == "attention-paper::chunk-0000"
    assert body["retrieval_results"] == _paper_evidence()
    assert llm.evidence == _paper_evidence()
    assert retriever.calls == [
        ("What is the Transformer based solely on?", "attention-paper")
    ]


def test_chat_returns_insufficient_evidence_without_calling_llm(monkeypatch) -> None:
    retriever = FakeRetriever([])
    llm = FakeLLM()
    monkeypatch.setattr(chat_api, "get_retriever", lambda: retriever)
    monkeypatch.setattr(chat_api, "get_groq_client", lambda: llm)

    response = TestClient(app).post(
        "/api/chat",
        json={"document_id": "attention-paper", "question": "Unknown fact?"},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "I don't have enough reliable evidence to answer this confidently."
    assert response.json()["sources"] == []
    assert response.json()["evidence"] == []
    assert llm.evidence is None


def test_chat_accepts_evidence_from_real_research_paper(monkeypatch) -> None:
    paper_path = Path(__file__).parent / "fixtures" / "attention_is_all_you_need.pdf"
    document = load_document(
        paper_path,
        document_id="real-attention-paper",
        source_filename="attention_is_all_you_need.pdf",
    )
    chunks = chunk_document(document)
    paper_chunk = next(
        chunk for chunk in chunks if "attention" in chunk.text.lower()
    )
    retriever = FakeRetriever([paper_chunk.__dict__ | {"score": 0.95}])
    llm = FakeLLM()
    monkeypatch.setattr(chat_api, "get_retriever", lambda: retriever)
    monkeypatch.setattr(chat_api, "get_groq_client", lambda: llm)

    response = TestClient(app).post(
        "/api/chat",
        json={
            "document_id": "real-attention-paper",
            "question": "What mechanism is central to the Transformer?",
        },
    )

    assert response.status_code == 200
    assert response.json()["document_id"] == "real-attention-paper"
    assert response.json()["retrieval_results"][0]["filename"] == "attention_is_all_you_need.pdf"
    assert llm.evidence[0]["document_id"] == "real-attention-paper"