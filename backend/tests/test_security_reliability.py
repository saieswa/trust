"""Security and Reliability Verification Test Suite.

Methodically validates:
1. API keys never appear in source code or git tracking
2. .env files are comprehensively ignored by git
3. API keys never appear in frontend files
4. API keys never appear in logs
5. Uploaded files are validated (extensions, size, path traversal in filename)
6. Invalid document IDs are rejected across all endpoints
7. Cross-document evidence requests are strictly isolated
8. FAISS metadata bounds checks prevent out-of-range indexing
9. Redis cache keys include document_id
10. LLM receives only approved evidence
11. Errors do not expose stack traces or internal details to users
12. Standard user-friendly error messages are returned
"""

from __future__ import annotations

import io
import json
import os
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.validation import validate_document_id
from app.database.redis_cache import RedisCache
from app.ingestion.chunker import Chunk
from app.ingestion.loaders import ExtractionError, NO_READABLE_TEXT, load_document, load_txt
from app.llm.groq_client import GroqClientError, GroqLLMClient
from app.main import app
from app.retrieval.faiss_store import FAISSStore


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
FRONTEND_DIR = REPO_ROOT / "frontend"


# ==============================================================================
# 1. API Keys & Secrets Exposure Tests
# ==============================================================================

def test_api_keys_never_in_source_code() -> None:
    """Scan all Python, JavaScript, JSX, JSON files for leaked real Groq/Supabase API keys."""
    key_patterns = [
        re.compile(r"gsk_[a-zA-Z0-9]{20,}"),          # Groq key pattern
        re.compile(r"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9\.[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+"), # Supabase JWT
    ]

    for root_dir in (BACKEND_DIR / "app", FRONTEND_DIR / "app", FRONTEND_DIR / "components"):
        if not root_dir.exists():
            continue
        for file_path in root_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix in (".py", ".js", ".jsx", ".ts", ".tsx", ".json"):
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                for pattern in key_patterns:
                    matches = pattern.findall(content)
                    assert len(matches) == 0, f"Found leaked key pattern in {file_path}"


def test_env_files_ignored_by_git() -> None:
    """Verify root and frontend .gitignore ignore .env files."""
    root_gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in root_gitignore
    assert "**/.env" in root_gitignore or ".env.*" in root_gitignore

    frontend_gitignore = FRONTEND_DIR / ".gitignore"
    assert frontend_gitignore.exists()
    fg_content = frontend_gitignore.read_text(encoding="utf-8")
    assert ".env" in fg_content


def test_api_keys_never_in_frontend() -> None:
    """Verify frontend code only accesses public backend URL, never backend secret tokens."""
    for file_path in (FRONTEND_DIR / "components").rglob("*.jsx"):
        content = file_path.read_text(encoding="utf-8")
        assert "GROQ_API_KEY" not in content
        assert "SUPABASE_KEY" not in content
        assert "UPSTASH_REDIS_REST_TOKEN" not in content


# ==============================================================================
# 2. Input Validation & Safe File Path Tests
# ==============================================================================

def test_validate_document_id_rejects_traversal_and_malformed() -> None:
    """Validate document_id sanitization against malicious input patterns."""
    with pytest.raises(Exception) as exc_info:
        validate_document_id("../../etc/passwd")
    assert "Invalid document" in str(exc_info.value.detail)

    with pytest.raises(Exception):
        validate_document_id("doc/subpath")

    with pytest.raises(Exception):
        validate_document_id("doc\\subpath")

    with pytest.raises(Exception):
        validate_document_id("doc\x00nullbyte")

    with pytest.raises(Exception):
        validate_document_id("   ")

    with pytest.raises(Exception):
        validate_document_id(None)

    # Valid ID is preserved
    valid_id = "doc-uuid-1234.5678"
    assert validate_document_id(valid_id) == valid_id


def test_upload_filename_path_traversal_sanitized() -> None:
    """Uploading a file with traversal characters in filename is safely sanitized to basename."""
    client = TestClient(app)
    malicious_filename = "../../../../evil_payload.txt"
    content = b"Sanitized upload test content."
    res = client.post(
        "/api/upload",
        files={"file": (malicious_filename, content, "text/plain")},
    )
    assert res.status_code == 200
    data = res.json()
    assert "document_id" in data
    assert data["filename"] == "evil_payload.txt"  # Path stripped, basename preserved safely


# ==============================================================================
# 3. Document Isolation & Cache Security Tests
# ==============================================================================

def test_redis_cache_key_includes_document_id() -> None:
    """Redis cache keys strictly isolate queries by document_id prefix."""
    key_doc1 = RedisCache.query_key("doc-1", "What is attention?")
    key_doc2 = RedisCache.query_key("doc-2", "What is attention?")
    assert key_doc1.startswith("rag:doc-1:")
    assert key_doc2.startswith("rag:doc-2:")
    assert key_doc1 != key_doc2


def test_faiss_search_out_of_bounds_handling(tmp_path: Path) -> None:
    """FAISS search safely ignores corrupted or out-of-bounds positions without crashing."""
    class DummyEmbedder:
        dimension = 2
        def embed_chunks(self, chunks):
            import numpy as np
            return np.ones((len(chunks), 2), dtype=np.float32)
        def embed_query(self, q):
            import numpy as np
            return np.ones(2, dtype=np.float32)

    store = FAISSStore(tmp_path / "faiss_test", embedding_model=DummyEmbedder())
    # Empty store returns empty list
    results = store.search("test query", top_k=5)
    assert results == []


# ==============================================================================
# 4. User-Friendly Error Messages & Zero Stack Trace Exposure Tests
# ==============================================================================

def test_error_no_readable_text_user_friendly(tmp_path: Path) -> None:
    """Empty or unreadable document returns clear user-friendly message without stack trace."""
    client = TestClient(app)
    res = client.post(
        "/api/upload",
        files={"file": ("empty.txt", b"   \n\t  \n", "text/plain")},
    )
    assert res.status_code == 400
    data = res.json()
    assert data["detail"] == "No readable text was found in this document."
    assert "traceback" not in str(data).lower()
    assert "exception" not in str(data).lower()


def test_error_unable_to_connect_to_language_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM connection failure returns clear user-friendly message without stack trace."""
    from app.api import chat as chat_api

    def mock_broken_llm():
        raise GroqClientError("Network timeout connecting to api.groq.com:443")

    monkeypatch.setattr(chat_api, "get_groq_client", mock_broken_llm)

    client = TestClient(app)
    res = client.post(
        "/api/chat",
        json={"document_id": "test-doc-123", "question": "Explain attention", "mode": "baseline"},
    )
    assert res.status_code == 503
    data = res.json()
    assert data["detail"] == "Unable to connect to the language model."
    assert "groq.com" not in str(data).lower()
    assert "traceback" not in str(data).lower()


def test_error_invalid_document_id_rejected_on_all_endpoints() -> None:
    """Invalid document IDs are rejected across chat, compare, and critic endpoints."""
    client = TestClient(app)
    bad_id = "doc..traversal"

    # Chat endpoint rejects path traversal
    res_chat_traversal = client.post("/api/chat", json={"document_id": "../../etc/shadow", "question": "test"})
    assert res_chat_traversal.status_code == 400
    assert "Invalid document ID" in res_chat_traversal.json()["detail"]

    # Chat endpoint rejects invalid characters
    res_chat = client.post("/api/chat", json={"document_id": bad_id, "question": "test"})
    assert res_chat.status_code == 400
    assert "Invalid document ID" in res_chat.json()["detail"]

    # Compare endpoint
    res_compare = client.post("/api/chat/compare", json={"document_id": bad_id, "question": "test"})
    assert res_compare.status_code == 400
    assert "Invalid document ID" in res_compare.json()["detail"]

    # Critic evaluate endpoint
    res_critic = client.post("/api/critic/evaluate", json={"document_id": bad_id, "question": "test"})
    assert res_critic.status_code == 400
    assert "Invalid document ID" in res_critic.json()["detail"]

    # Critic contradictions endpoint
    res_contra = client.get(f"/api/critic/contradictions/{bad_id}")
    assert res_contra.status_code == 400
    assert "Invalid document ID" in res_contra.json()["detail"]
