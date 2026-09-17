"""Strict upload-to-chat document isolation test."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from app.api import upload as upload_api
from app.api import chat as chat_api
from app.database.supabase import SupabaseDatabase
from app.llm.groq_client import FALLBACK_ANSWER
from app.main import app
from app.retrieval.faiss_store import FAISSStore
from app.retrieval.retriever import Retriever


class FakeEmbeddingModel:
    dimension = 4

    def _vector(self, text):
        text = text.lower()
        if "paper a" in text or "attention" in text:
            return np.array([1, 0, 0, 1], dtype=np.float32)
        if "paper b" in text or "bert" in text:
            return np.array([0, 1, 0, 1], dtype=np.float32)
        if "title" in text or "authors" in text:
            return np.array([0, 0, 0, 1], dtype=np.float32)
        return np.array([0, 0, 1, 0], dtype=np.float32)

    def embed_chunks(self, chunks):
        return np.vstack([self._vector(chunk.text) for chunk in chunks])

    def embed_query(self, question):
        return self._vector(question)


class FakeLLM:
    def answer(self, question, evidence):
        text = evidence[0]["text"]
        if "title" in question.lower():
            answer = text.split("Title: ", 1)[1].split(".", 1)[0]
        elif "authors" in question.lower():
            answer = text.split("Authors: ", 1)[1].split(".", 1)[0]
        else:
            answer = text
        return {
            "answer": answer,
            "source_references": [{"source": evidence[0]["source"], "page_number": 1}],
            "evidence_references": [{"chunk_id": evidence[0]["chunk_id"], "document_id": evidence[0]["document_id"], "source": evidence[0]["source"], "page_number": 1}],
        }


class FakeSupabase:
    def __init__(self):
        self.database = SupabaseDatabase(client=FakeClient())


class FakeClient:
    def __init__(self):
        self.tables = {"documents": [], "chunks": []}

    def table(self, name):
        return FakeTable(self.tables[name])


class FakeTable:
    def __init__(self, records):
        self.records = records
        self.payload = None

    def insert(self, payload):
        self.payload = payload
        return self

    def execute(self):
        inserted = self.payload if isinstance(self.payload, list) else [self.payload]
        self.records.extend(inserted)
        return type("Response", (), {"data": inserted})()


def test_two_uploaded_papers_never_cross_contaminate(tmp_path, monkeypatch):
    embedder = FakeEmbeddingModel()
    store = FAISSStore(tmp_path / "faiss", embedding_model=embedder)
    database = FakeSupabase().database
    monkeypatch.setattr(upload_api, "uploads_dir", str(tmp_path / "uploads"))
    monkeypatch.setattr(upload_api, "get_database", lambda: database)
    monkeypatch.setattr(upload_api, "get_vector_store", lambda: store)
    monkeypatch.setattr(chat_api, "get_retriever", lambda: Retriever(store=store, embedding_model=embedder, relevance_threshold=0.6))
    monkeypatch.setattr(chat_api, "get_groq_client", lambda: FakeLLM())
    client = TestClient(app)

    paper_a_upload = client.post(
        "/api/upload",
        files={"file": ("paper_a.txt", "Title: Attention Is All You Need. Authors: Ashish Vaswani et al. Paper A uses attention.", "text/plain")},
    )
    paper_b_upload = client.post(
        "/api/upload",
        files={"file": ("paper_b.txt", "Title: BERT. Authors: Jacob Devlin et al. Paper B uses masked language modeling.", "text/plain")},
    )
    assert paper_a_upload.status_code == 200
    assert paper_b_upload.status_code == 200
    document_a = paper_a_upload.json()["document_id"]
    document_b = paper_b_upload.json()["document_id"]

    title_a = client.post("/api/chat", json={"document_id": document_a, "question": "What is the title of this paper?"})
    title_b = client.post("/api/chat", json={"document_id": document_b, "question": "What is the title of this paper?"})
    authors_b = client.post("/api/chat", json={"document_id": document_b, "question": "Who are the authors of this paper?"})
    cross_document = client.post("/api/chat", json={"document_id": document_b, "question": "What is the attention mechanism used in Paper A?"})

    assert title_a.json()["answer"] == "Attention Is All You Need"
    assert title_b.json()["answer"] == "BERT"
    assert authors_b.json()["answer"] == "Jacob Devlin et al"
    cross_json = cross_document.json()
    cross_json.pop("performance", None)
    assert cross_json == {
        "answer": FALLBACK_ANSWER,
        "document_id": document_b,
        "sources": [],
        "evidence": [],
        "retrieval_results": [],
        "mode": "baseline",
    }