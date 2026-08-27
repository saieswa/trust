from dataclasses import dataclass
from typing import List

from app.ingestion.loaders import ExtractedDocument


# Reasonable starting values for semantic retrieval.
# These can be tuned later using retrieval evaluation.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


@dataclass
class DocumentChunk:
    """
    Structured representation of a retrieval chunk.
    """

    document_id: str
    chunk_id: str
    filename: str
    page_number: int | str | None
    text: str
    source: str


def _split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping character-based chunks.

    The splitter tries to break at natural boundaries such as
    paragraphs, sentences, or whitespace instead of cutting
    words in half.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    text = text.strip()

    if not text:
        return []

    chunks: List[str] = []

    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            boundary = text.rfind("\n\n", start, end)

            if boundary <= start:
                boundary = text.rfind(". ", start, end)

            if boundary <= start:
                boundary = text.rfind(" ", start, end)

            if boundary > start:
                end = boundary + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = end - chunk_overlap

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def chunk_documents(
    extracted_documents: List[ExtractedDocument],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[DocumentChunk]:
    """
    Convert extracted document content into structured chunks.

    Each extracted document is processed independently.

    Chunks from different documents are never mixed.
    Every chunk retains the original document_id.
    """

    chunks: List[DocumentChunk] = []

    for extracted in extracted_documents:

        text_chunks = _split_text(
            text=extracted.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for index, chunk_text in enumerate(
            text_chunks,
            start=1,
        ):

            chunk_id = (
                f"{extracted.document_id}"
                f"-chunk-{index:04d}"
            )

            source = (
                f"{extracted.filename}"
                f"#page={extracted.page}"
            )

            chunks.append(
                DocumentChunk(
                    document_id=extracted.document_id,
                    chunk_id=chunk_id,
                    filename=extracted.filename,
                    page_number=extracted.page,
                    text=chunk_text,
                    source=source,
                )
            )

    return chunks