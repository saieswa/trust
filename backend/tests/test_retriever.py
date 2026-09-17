"""Tests for document-scoped retrieval."""

from __future__ import annotations

import numpy as np

from app.ingestion.chunker import Chunk
from app.retrieval.faiss_store import FAISSStore
from app.retrieval.retriever import Retriever


class PaperEmbeddingModel:
    dimension = 3

    def _vector(self, text: str) -> np.ndarray:
        lowered = text.lower()
        if "attention" in lowered or "transformer" in lowered:
            return np.array([1, 0, 0], dtype=np.float32)
        if "convolution" in lowered or "image" in lowered:
            return np.array([0, 1, 0], dtype=np.float32)
        return np.array([0, 0, 1], dtype=np.float32)

    def embed_chunks(self, chunks):
        return np.vstack([self._vector(chunk.text) for chunk in chunks])

    def embed_query(self, question):
        return self._vector(question)


def _paper_chunk(document_id: str, text: str) -> Chunk:
    filename = f"{document_id}.pdf"
    return Chunk(
        document_id=document_id,
        chunk_id=f"{document_id}::chunk-0000",
        filename=filename,
        page_number=1,
        text=text,
        source=f"{filename}#page=1",
    )


def _retriever(tmp_path):
    embedder = PaperEmbeddingModel()
    store = FAISSStore(tmp_path, embedding_model=embedder)
    store.add_chunks(
        [
            _paper_chunk("paper-a", "Attention and transformer architecture."),
            _paper_chunk("paper-b", "Convolutional image recognition."),
        ]
    )
    return Retriever(store=store, embedding_model=embedder, relevance_threshold=0.5)


def test_retrieves_evidence_from_paper_a(tmp_path) -> None:
    results = _retriever(tmp_path).retrieve(
        "How does the transformer work?", document_id="paper-a"
    )

    assert results
    assert all(result["document_id"] == "paper-a" for result in results)
    assert results[0]["page_number"] == 1
    assert results[0]["score"] >= 0.5


def test_paper_b_question_cannot_return_paper_a(tmp_path) -> None:
    results = _retriever(tmp_path).retrieve(
        "How does image recognition use convolution?", document_id="paper-b"
    )

    assert results
    assert all(result["document_id"] == "paper-b" for result in results)
    assert all("paper-a" not in result["source"] for result in results)


def test_returns_no_evidence_for_missing_answer(tmp_path) -> None:
    results = _retriever(tmp_path).retrieve(
        "What is the boiling point of water?", document_id="paper-b"
    )

    assert results == []