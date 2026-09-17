"""Tests for persistent, document-scoped FAISS retrieval."""

from __future__ import annotations

import numpy as np

from app.ingestion.chunker import Chunk
from app.retrieval.faiss_store import FAISSStore


class StubEmbeddingModel:
    dimension = 3

    def embed_chunks(self, chunks):
        return np.ones((len(chunks), self.dimension), dtype=np.float32)

    def embed_query(self, question):
        return np.array([1, 0, 0], dtype=np.float32)


def _chunk(document_id: str, index: int) -> Chunk:
    filename = f"{document_id}.pdf"
    return Chunk(
        document_id=document_id,
        chunk_id=f"{document_id}::chunk-{index:04d}",
        filename=filename,
        page_number=1,
        text=f"Content from {document_id}.",
        source=f"{filename}#page=1",
    )


def test_indexes_two_papers_incrementally_and_isolates_results(tmp_path) -> None:
    embedder = StubEmbeddingModel()
    store = FAISSStore(tmp_path, embedding_model=embedder)
    paper_a = _chunk("paper-a", 0)
    paper_b = _chunk("paper-b", 0)

    assert store.add_chunks([paper_a], np.array([[1, 0, 0]], dtype=np.float32)) == 1
    assert store.add_chunks([paper_b], np.array([[0.9, 0.1, 0]], dtype=np.float32)) == 1
    assert store.size == 2
    assert (tmp_path / "index.faiss").exists()
    assert (tmp_path / "metadata.json").exists()

    reloaded = FAISSStore(tmp_path, embedding_model=embedder)
    results = reloaded.search(np.array([1, 0, 0], dtype=np.float32), top_k=5, document_id="paper-b")

    assert len(results) == 1
    assert results[0]["document_id"] == "paper-b"
    assert results[0]["chunk_id"] == paper_b.chunk_id
    assert set(results[0]) >= {
        "document_id",
        "chunk_id",
        "filename",
        "page_number",
        "text",
        "source",
        "score",
    }