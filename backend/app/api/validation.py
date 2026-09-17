"""Centralized input validation helpers for Trust-Aware RAG API."""

from __future__ import annotations

import re
from fastapi import HTTPException, status

# Allowed document_id characters: alphanumeric, dashes, underscores, dots (without ..)
DOC_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.:]+$")


def validate_document_id(document_id: str | None) -> str:
    """Validate and sanitize a user-supplied document_id.

    Rejects empty values, path traversal sequences, illegal path characters,
    and null bytes with clear, secure HTTP exceptions.
    """
    if document_id is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="document_id must contain non-empty text",
        )

    cleaned = document_id.strip()
    if not cleaned:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="document_id must contain non-empty text",
        )

    if len(cleaned) > 128:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document_id: exceeds maximum allowed length.",
        )

    if ".." in cleaned or "/" in cleaned or "\\" in cleaned or "\x00" in cleaned:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID / Invalid document_id: contains illegal characters or path traversal sequences.",
        )

    if not DOC_ID_PATTERN.match(cleaned):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID / Invalid document_id: contains unsupported characters.",
        )

    return cleaned
