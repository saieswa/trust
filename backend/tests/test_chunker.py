"""Tests for document chunking.

Does not send chunks to Groq and does not create embeddings.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    ChunkingError,
    chunk_document,
    split_text,
)
from app.ingestion.loaders import ExtractedDocument, ExtractedPage, load_document

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PAPER_PDF = FIXTURES / "attention_is_all_you_need.pdf"
PAPER_URL = "https://arxiv.org/pdf/1706.03762"


@pytest.fixture(scope="session")
def research_paper_pdf() -> Path:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    if PAPER_PDF.exists() and PAPER_PDF.stat().st_size > 10_000:
        return PAPER_PDF

    import urllib.request

    request = urllib.request.Request(
        PAPER_URL,
        headers={"User-Agent": "trust-aware-rag-tests/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
    except Exception as exc:
        pytest.fail(f"Failed to download research paper PDF from {PAPER_URL}: {exc}")

    if len(data) < 10_000 or not data.startswith(b"%PDF"):
        pytest.fail("Downloaded research paper is not a valid PDF.")

    PAPER_PDF.write_bytes(data)
    return PAPER_PDF


def _print_chunk_report(chunks) -> None:
    print(f"total_chunks={len(chunks)}")
    print(f"first_chunk_ids={[chunk.chunk_id for chunk in chunks[:5]]}")
    print(f"page_numbers={[chunk.page_number for chunk in chunks[:12]]}")
    print(
        "char_counts="
        + str([chunk.char_count for chunk in chunks[:12]])
    )
    print(
        "word_counts="
        + str([chunk.word_count for chunk in chunks[:12]])
    )
    print(
        "chunk_size_config="
        f"{DEFAULT_CHUNK_SIZE} overlap={DEFAULT_CHUNK_OVERLAP}"
    )


def test_research_paper_chunking_reports_metadata(research_paper_pdf: Path) -> None:
    document_id = "doc-attention-paper"
    extracted = load_document(
        research_paper_pdf,
        document_id=document_id,
        source_filename="attention_is_all_you_need.pdf",
    )
    chunks = chunk_document(extracted)
    _print_chunk_report(chunks)

    assert chunks
    assert all(chunk.document_id == document_id for chunk in chunks)
    assert all(chunk.filename == "attention_is_all_you_need.pdf" for chunk in chunks)
    assert all(chunk.page_number is not None for chunk in chunks)
    assert all(chunk.text.strip() for chunk in chunks)
    assert all(chunk.source.startswith("attention_is_all_you_need.pdf#page=") for chunk in chunks)
    assert chunks[0].chunk_id == f"{document_id}::chunk-0000"
    assert [chunk.chunk_id for chunk in chunks] == [
        f"{document_id}::chunk-{index:04d}" for index in range(len(chunks))
    ]
    assert all(chunk.char_count <= DEFAULT_CHUNK_SIZE + DEFAULT_CHUNK_OVERLAP for chunk in chunks)
    assert len(chunks) >= extracted.page_count


def test_chunks_from_different_documents_are_not_mixed() -> None:
    first = ExtractedDocument(
        document_id="doc-a",
        source_filename="a.txt",
        file_type="txt",
        pages=(
            ExtractedPage(
                document_id="doc-a",
                source_filename="a.txt",
                page_number=1,
                text="Alpha document content about retrieval.",
            ),
        ),
    )
    second = ExtractedDocument(
        document_id="doc-b",
        source_filename="b.txt",
        file_type="txt",
        pages=(
            ExtractedPage(
                document_id="doc-b",
                source_filename="b.txt",
                page_number=1,
                text="Beta document content about verification.",
            ),
        ),
    )

    chunks_a = chunk_document(first)
    chunks_b = chunk_document(second)

    assert {chunk.document_id for chunk in chunks_a} == {"doc-a"}
    assert {chunk.document_id for chunk in chunks_b} == {"doc-b"}
    assert not set(chunk.chunk_id for chunk in chunks_a) & set(
        chunk.chunk_id for chunk in chunks_b
    )


def test_rejects_pages_from_multiple_documents() -> None:
    mixed = ExtractedDocument(
        document_id="doc-a",
        source_filename="mixed.txt",
        file_type="txt",
        pages=(
            ExtractedPage("doc-a", "mixed.txt", 1, "Page from A."),
            ExtractedPage("doc-b", "mixed.txt", 2, "Page from B."),
        ),
    )
    with pytest.raises(ChunkingError, match="must not be mixed"):
        chunk_document(mixed)


def test_split_text_uses_overlap() -> None:
    text = " ".join(f"word{i:03d}" for i in range(80))
    chunks = split_text(text, chunk_size=120, chunk_overlap=30)
    assert len(chunks) >= 2
    assert all(len(chunk) <= 120 for chunk in chunks)
    assert chunks[0][:20] in chunks[0]
    overlap_window = chunks[0][-30:]
    assert overlap_window.strip()
    assert any(token in chunks[1] for token in overlap_window.split() if token)


def test_invalid_overlap_raises() -> None:
    document = ExtractedDocument(
        document_id="doc-x",
        source_filename="x.txt",
        file_type="txt",
        pages=(ExtractedPage("doc-x", "x.txt", 1, "Some readable text."),),
    )
    with pytest.raises(ChunkingError, match="chunk_overlap must be smaller"):
        chunk_document(document, chunk_size=100, chunk_overlap=100)
