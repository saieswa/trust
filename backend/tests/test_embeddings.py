"""Tests for normalized sentence-transformer embeddings."""

from __future__ import annotations

import numpy as np

from app.embeddings.embedding_model import EmbeddingModel
from app.ingestion.chunker import Chunk


def test_embeds_chunk_and_question_with_shared_dimension() -> None:
    embedder = EmbeddingModel()
    chunk = Chunk(
        document_id="doc-test",
        chunk_id="doc-test::chunk-0000",
        filename="research-paper.pdf",
        page_number=1,
        text="Semantic retrieval compares normalized document and query vectors.",
        source="research-paper.pdf#page=1",
    )

    chunk_vector = embedder.embed_chunks([chunk])
    query_vector = embedder.embed_query("How does semantic retrieval compare text?")

    print(f"embedding_dimension={embedder.dimension}")
    print(f"chunk_shape={chunk_vector.shape} query_shape={query_vector.shape}")

    assert embedder.dimension == 384
    assert chunk_vector.shape == (1, 384)
    assert query_vector.shape == (384,)
    assert np.isclose(np.linalg.norm(chunk_vector[0]), 1.0, atol=1e-5)
    assert np.isclose(np.linalg.norm(query_vector), 1.0, atol=1e-5)