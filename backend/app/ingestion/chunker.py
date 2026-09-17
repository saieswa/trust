"""Split extracted document pages into retrieval-ready chunks."""

from __future__ import annotations

from dataclasses import dataclass
import re

from app.ingestion.loaders import ExtractedDocument

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


class ChunkingError(ValueError):
    """Raised when extracted content cannot be chunked safely."""


@dataclass(frozen=True)
class Chunk:
    document_id: str
    chunk_id: str
    filename: str
    page_number: int | str | None
    text: str
    source: str

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(re.findall(r"\S+", self.text))


DocumentChunk = Chunk


def _find_boundary(text: str, start: int, end: int) -> int:
    boundaries = ("\n\n", ". ", "? ", "! ", " ")
    candidates = [text.rfind(boundary, start, end) for boundary in boundaries]
    boundary = max(candidates)
    if boundary <= start:
        return end
    return boundary + (2 if text[boundary : boundary + 2] in {"\n\n", ". ", "? ", "! "} else 1)


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into naturally bounded, character-sized overlapping chunks."""
    if chunk_size <= 0:
        raise ChunkingError("chunk_size must be greater than 0.")
    if chunk_overlap < 0:
        raise ChunkingError("chunk_overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ChunkingError("chunk_overlap must be smaller than chunk_size.")

    normalized = text.strip()
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        proposed_end = min(start + chunk_size, len(normalized))
        end = proposed_end if proposed_end == len(normalized) else _find_boundary(
            normalized, start, proposed_end
        )
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(end - chunk_overlap, start + 1)

    return chunks


def chunk_document(
    document: ExtractedDocument,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Chunk one extracted document while preserving page and document metadata."""
    document_ids = {page.document_id for page in document.pages}
    if document.document_id not in document_ids:
        raise ChunkingError("Every page must retain the document's document_id.")
    if len(document_ids) > 1:
        raise ChunkingError("Extracted document pages must not be mixed.")

    chunks: list[Chunk] = []
    for page in document.pages:
        for text in split_text(page.text, chunk_size, chunk_overlap):
            chunk_index = len(chunks)
            chunks.append(
                Chunk(
                    document_id=document.document_id,
                    chunk_id=f"{document.document_id}::chunk-{chunk_index:04d}",
                    filename=document.source_filename,
                    page_number=page.page_number,
                    text=text,
                    source=f"{document.source_filename}#page={page.page_number}",
                )
            )
    return chunks


def chunk_documents(
    documents: list[ExtractedDocument],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Chunk multiple documents independently without sharing chunk state."""
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size, chunk_overlap))
    return chunks
