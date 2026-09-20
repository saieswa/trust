"""Split extracted document pages into retrieval-ready, sentence-aware chunks."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from app.ingestion.loaders import ExtractedDocument

DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 180


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
    page: int | str | None = None
    canonical_source: str | None = None

    def __post_init__(self):
        if self.page is None and self.page_number is not None:
            object.__setattr__(self, "page", self.page_number)
        if self.canonical_source is None:
            clean_name = self.filename.split("#")[0].strip() if self.filename else "Document"
            cs = f"{clean_name} — Page {self.page_number}" if self.page_number is not None else clean_name
            object.__setattr__(self, "canonical_source", cs)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(re.findall(r"\S+", self.text))

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "filename": self.filename,
            "page": self.page_number,
            "page_number": self.page_number,
            "text": self.text,
            "source": self.source,
            "canonical_source": self.canonical_source,
        }


DocumentChunk = Chunk


def _find_boundary(text: str, start: int, end: int) -> int:
    """Find the best natural ending boundary for a chunk."""
    for b in ("\n\n", ". ", "? ", "! "):
        cand = text.rfind(b, start, end)
        if cand > start:
            return cand + len(b)
    cand = text.rfind(" ", start, end)
    if cand > start:
        return cand + 1
    return end


def _find_start_boundary(text: str, start: int, end: int) -> int:
    """Find the best natural starting boundary for an overlapping chunk."""
    for b in ("\n\n", ". ", "? ", "! "):
        cand = text.rfind(b, start, end)
        if cand > start:
            return cand + len(b)
    cand = text.rfind(" ", start, end)
    if cand > start:
        return cand + 1
    return start


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into naturally bounded, sentence-preserving overlapping chunks."""
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
        end = proposed_end if proposed_end == len(normalized) else _find_boundary(normalized, start, proposed_end)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        target_start = max(end - chunk_overlap, start + 1)
        start = _find_start_boundary(normalized, start + 1, target_start)
        if start >= end:
            start = end

    return chunks


def _format_canonical_source(filename: str, page_number: int | str | None) -> str:
    """Format canonical user-facing source without '#page=' clutter."""
    clean_name = filename.split("#")[0].strip() if filename else "Document"
    if page_number is not None and str(page_number).strip():
        return f"{clean_name} — Page {page_number}"
    return clean_name


def chunk_document(
    document: ExtractedDocument,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Chunk one extracted document while preserving sentence boundaries and metadata."""
    document_ids = {page.document_id for page in document.pages}
    if document.document_id not in document_ids:
        raise ChunkingError("Every page must retain the document's document_id.")
    if len(document_ids) > 1:
        raise ChunkingError("Extracted document pages must not be mixed.")

    chunks: list[Chunk] = []
    for page in document.pages:
        page_chunks = split_text(page.text, chunk_size, chunk_overlap)
        canonical_source = _format_canonical_source(document.source_filename, page.page_number)
        raw_source = f"{document.source_filename}#page={page.page_number}"
        for text in page_chunks:
            chunk_index = len(chunks)
            chunks.append(
                Chunk(
                    document_id=document.document_id,
                    chunk_id=f"{document.document_id}::chunk-{chunk_index:04d}",
                    filename=document.source_filename,
                    page_number=page.page_number,
                    page=page.page_number,
                    text=text,
                    source=raw_source,
                    canonical_source=canonical_source,
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

