"""Upload API — large-document-safe pipeline.

Design:
  POST /api/upload  → stream file to disk → return {document_id, job_id} immediately.
  Background task   → extract → chunk → embed (batched) → FAISS (batched) → persist.
  GET  /api/upload/status/{job_id} → poll for stage / progress / completion / error.

File-size limit: configurable via MAX_FILE_SIZE_MB env var (default 500 MB).
Embedding batch:  configurable via EMBEDDING_BATCH_SIZE env var (default 64 chunks).
DB insert batch:  100 rows per Supabase call.
"""

import asyncio
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.database.metadata import save_document_metadata
from app.database.supabase import SupabaseDatabaseError, get_database
from app.embeddings.embedding_model import EmbeddingError
from app.ingestion.chunker import ChunkingError, chunk_document
from app.ingestion.loaders import ExtractionError, load_document
from app.retrieval.faiss_store import FAISSStore, FAISSStoreError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

SUPPORTED_EXTENSIONS = {"pdf", "txt", "doc", "docx", "csv", "json", "url"}

# File-size cap — raised from 25 MB to 500 MB.  Override via MAX_FILE_SIZE_MB env var.
_env_max_mb = int(os.getenv("MAX_FILE_SIZE_MB", "500"))
MAX_FILE_SIZE_BYTES = _env_max_mb * 1024 * 1024

# Embedding / FAISS batch size.  64 chunks ≈ 64 × 900 chars ≈ safe RAM budget.
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

# Supabase insert batch — never send more than 100 rows per call.
DB_INSERT_BATCH = 100

# Base directory for the backend
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
uploads_dir = os.path.join(backend_dir, "data", "uploads")

# Ensure uploads directory exists
os.makedirs(uploads_dir, exist_ok=True)

# In-memory job store  {job_id -> status dict}
# Keys: status, stage, progress_pct, chunks_done, chunks_total, error,
#       document_id, filename, started_at, finished_at
_job_store: dict[str, dict] = {}

# Dedicated thread-pool for CPU-bound PDF/embedding work
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="indexer")

_vector_store: FAISSStore | None = None


def get_vector_store() -> FAISSStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = FAISSStore()
    return _vector_store


# ---------------------------------------------------------------------------
# Background indexing task
# ---------------------------------------------------------------------------

def _run_indexing(job_id: str, file_path: str, document_id: str, filename: str, ext: str) -> None:
    """CPU-bound indexing pipeline — runs in the thread executor."""
    job = _job_store[job_id]

    def _update(stage: str, pct: int, chunks_done: int = 0, chunks_total: int = 0) -> None:
        job.update(
            stage=stage,
            progress_pct=pct,
            chunks_done=chunks_done,
            chunks_total=chunks_total,
        )

    try:
        # ── 1. Record "uploaded" status in Supabase (best-effort) ──────────
        try:
            database = get_database()
            if database.configured:
                database.insert_document(
                    document_id=document_id,
                    filename=filename,
                    file_type=ext,
                    source=file_path,
                    processing_status="indexing",
                    upload_date=datetime.utcnow(),
                )
        except Exception as db_exc:
            logger.warning("[UPLOAD] Supabase insert_document failed (non-fatal): %s", db_exc)

        # ── 2. Extract text ────────────────────────────────────────────────
        _update("Extracting pages", 10)
        extracted = load_document(file_path, document_id=document_id, source_filename=filename)
        total_chars = sum(len(p.text) for p in extracted.pages)
        logger.info(
            "[INGESTION] document_id=%s pages=%d total_chars=%d",
            document_id, extracted.page_count, total_chars,
        )

        # ── 3. Chunk ───────────────────────────────────────────────────────
        _update("Chunking text", 20)
        chunks = chunk_document(extracted)
        total_chunks = len(chunks)
        logger.info("[CHUNKING] document_id=%s chunks=%d", document_id, total_chunks)
        job["chunks_total"] = total_chunks

        # ── 4. Embed + FAISS (batched) ─────────────────────────────────────
        store = get_vector_store()
        vectors_before = store.size
        logger.info(
            "[EMBEDDING] document_id=%s batch_size=%d chunks=%d",
            document_id, EMBEDDING_BATCH_SIZE, total_chunks,
        )

        chunks_done = 0
        for done, total in store.add_chunks_batched(chunks, batch_size=EMBEDDING_BATCH_SIZE):
            chunks_done = done
            # Scale progress from 25% → 90% during embedding/indexing
            embed_pct = 25 + int(65 * done / max(total, 1))
            _update("Embedding & indexing", embed_pct, done, total)

        logger.info(
            "[FAISS] document_id=%s vectors_before=%d vectors_after=%d",
            document_id, vectors_before, store.size,
        )

        # ── 5. Save local metadata ─────────────────────────────────────────
        _update("Saving metadata", 92, chunks_done, total_chunks)
        try:
            save_document_metadata(
                document_id,
                {
                    "document_id": document_id,
                    "filename": filename,
                    "file_type": ext,
                    "upload_date": datetime.utcnow().isoformat(),
                    "chunks_count": total_chunks,
                    "page_count": extracted.page_count,
                },
            )
        except Exception as save_err:
            logger.warning("[UPLOAD] Failed saving local metadata (non-fatal): %s", save_err)

        # ── 6. Supabase chunk inserts (batched, best-effort) ───────────────
        _update("Persisting to database", 94, chunks_done, total_chunks)
        try:
            database = get_database()
            if database.configured:
                chunk_records = [
                    {
                        "chunk_id": c.chunk_id,
                        "document_id": c.document_id,
                        "filename": c.filename,
                        "page_number": c.page_number,
                        "text": c.text,
                        "source": c.source,
                        "metadata": {
                            "source": c.source,
                            "char_count": c.char_count,
                            "word_count": c.word_count,
                        },
                    }
                    for c in chunks
                ]
                for i in range(0, len(chunk_records), DB_INSERT_BATCH):
                    batch = chunk_records[i : i + DB_INSERT_BATCH]
                    try:
                        database.insert_chunks(batch)
                    except Exception as batch_exc:
                        logger.warning(
                            "[UPLOAD] Supabase insert_chunks batch %d failed (non-fatal): %s",
                            i // DB_INSERT_BATCH, batch_exc,
                        )
        except Exception as db_exc:
            logger.warning("[UPLOAD] Supabase chunk insert skipped (non-fatal): %s", db_exc)

        # ── 7. Done ────────────────────────────────────────────────────────
        job.update(
            status="completed",
            stage="Completed",
            progress_pct=100,
            chunks_done=total_chunks,
            chunks_total=total_chunks,
            finished_at=datetime.utcnow().isoformat(),
        )
        logger.info("[UPLOAD] document_id=%s indexing completed, chunks=%d", document_id, total_chunks)

    except ExtractionError as exc:
        err_msg = str(exc).lower()
        detail = (
            "No readable text was found in this document."
            if ("readable text" in err_msg or "empty" in err_msg)
            else f"Document extraction failed: {exc}"
        )
        logger.error("[UPLOAD] ExtractionError document_id=%s: %s", document_id, exc)
        job.update(
            status="failed",
            stage="Failed",
            error=detail,
            finished_at=datetime.utcnow().isoformat(),
        )
        _cleanup_file(file_path)

    except (ChunkingError, EmbeddingError, FAISSStoreError, SupabaseDatabaseError) as exc:
        detail = f"{type(exc).__name__}: {exc}"
        logger.exception("[UPLOAD] Pipeline error document_id=%s: %s", document_id, exc)
        job.update(
            status="failed",
            stage="Failed",
            error=detail,
            finished_at=datetime.utcnow().isoformat(),
        )
        _cleanup_file(file_path)

    except Exception as exc:
        detail = f"Unexpected error during indexing: {type(exc).__name__}: {exc}"
        logger.exception("[UPLOAD] Unexpected error document_id=%s: %s", document_id, exc)
        job.update(
            status="failed",
            stage="Failed",
            error=detail,
            finished_at=datetime.utcnow().isoformat(),
        )
        _cleanup_file(file_path)


def _cleanup_file(file_path: str) -> None:
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as exc:
        logger.warning("[UPLOAD] Failed to delete temp file %s: %s", file_path, exc)


# ---------------------------------------------------------------------------
# POST /api/upload — receive file, start background job
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Stream file to disk, create background indexing job, return immediately.

    The caller should poll GET /api/upload/status/{job_id} for progress.
    Large PDFs (300+ pages) are fully supported; processing is batched.
    """
    # ── 1. Validate filename ──────────────────────────────────────────────
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No filename provided")

    clean_filename = os.path.basename(filename.replace("\\", "/"))
    if not clean_filename or clean_filename in (".", ".."):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid filename provided.")

    _, ext_raw = os.path.splitext(clean_filename)
    ext = ext_raw.lower().lstrip(".")
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: .{ext}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    # ── 2. Generate IDs ───────────────────────────────────────────────────
    document_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())

    # ── 3. Stream file to disk ────────────────────────────────────────────
    safe_filename = f"{document_id}_{clean_filename}"
    file_path = os.path.abspath(os.path.join(uploads_dir, safe_filename))
    uploads_dir_abs = os.path.abspath(uploads_dir)
    if not file_path.startswith(uploads_dir_abs):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid filename.")

    try:
        os.makedirs(uploads_dir, exist_ok=True)
        total_bytes = 0
        read_chunk_size = 1024 * 1024  # 1 MB read buffer
        with open(file_path, "wb") as buf:
            while True:
                data = await file.read(read_chunk_size)
                if not data:
                    break
                total_bytes += len(data)
                if total_bytes > MAX_FILE_SIZE_BYTES:
                    buf.close()
                    _cleanup_file(file_path)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(
                            f"File exceeds the maximum allowed size of {_env_max_mb} MB. "
                            f"Set MAX_FILE_SIZE_MB env var to increase the limit."
                        ),
                    )
                buf.write(data)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[UPLOAD] Failed to save file to disk: %s", e)
        _cleanup_file(file_path)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not save uploaded file.")

    logger.info(
        "[UPLOAD] file saved: document_id=%s filename=%s size_bytes=%d",
        document_id, clean_filename, total_bytes,
    )

    # ── 4. Register job ───────────────────────────────────────────────────
    _job_store[job_id] = {
        "job_id": job_id,
        "document_id": document_id,
        "filename": clean_filename,
        "status": "indexing",
        "stage": "Queued",
        "progress_pct": 5,
        "chunks_done": 0,
        "chunks_total": 0,
        "error": None,
        "started_at": datetime.utcnow().isoformat(),
        "finished_at": None,
    }

    # ── 5. Start background indexing (non-blocking) ───────────────────────
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        _executor,
        _run_indexing,
        job_id,
        file_path,
        document_id,
        filename,
        ext,
    )

    return {
        "document_id": document_id,
        "job_id": job_id,
        "filename": clean_filename,
        "status": "indexing",
        "message": "File uploaded. Indexing has started in the background. Poll /api/upload/status/{job_id} for progress.",
    }


# ---------------------------------------------------------------------------
# GET /api/upload/status/{job_id} — poll indexing progress
# ---------------------------------------------------------------------------

@router.get("/upload/status/{job_id}")
async def upload_status(job_id: str):
    """Return the current indexing progress for a previously uploaded document.

    Clients should poll this endpoint every 2 seconds until status is
    'completed' or 'failed'.
    """
    job = _job_store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No indexing job found for job_id '{job_id}'. The server may have restarted.",
        )
    return dict(job)


@router.get("/upload/jobs")
async def list_upload_jobs():
    """Return all active and recent indexing jobs."""
    return list(_job_store.values())


# ---------------------------------------------------------------------------
# POST /api/upload/url — URL ingestion (unchanged logic, same improvements)
# ---------------------------------------------------------------------------

class UrlUploadRequest(BaseModel):
    url: str = Field(min_length=1)


@router.post("/upload/url")
async def upload_url_document(request: UrlUploadRequest):
    """Ingest web page content directly from a URL."""
    url = request.url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid URL: must start with http:// or https://",
        )

    document_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())

    _job_store[job_id] = {
        "job_id": job_id,
        "document_id": document_id,
        "filename": url,
        "status": "indexing",
        "stage": "Fetching URL",
        "progress_pct": 5,
        "chunks_done": 0,
        "chunks_total": 0,
        "error": None,
        "started_at": datetime.utcnow().isoformat(),
        "finished_at": None,
    }

    def _run_url():
        job = _job_store[job_id]
        try:
            job.update(stage="Fetching URL", progress_pct=10)
            extracted = load_document(url, document_id=document_id)
            job.update(stage="Chunking", progress_pct=25)
            chunks = chunk_document(extracted)
            total = len(chunks)
            job["chunks_total"] = total

            store = get_vector_store()
            for done, t in store.add_chunks_batched(chunks, batch_size=EMBEDDING_BATCH_SIZE):
                pct = 30 + int(60 * done / max(t, 1))
                job.update(stage="Embedding & indexing", progress_pct=pct, chunks_done=done, chunks_total=t)

            try:
                save_document_metadata(
                    document_id,
                    {
                        "document_id": document_id,
                        "filename": url,
                        "file_type": "url",
                        "upload_date": datetime.utcnow().isoformat(),
                        "chunks_count": total,
                    },
                )
            except Exception as e:
                logger.warning("[UPLOAD_URL] local metadata save failed (non-fatal): %s", e)

            try:
                database = get_database()
                if database.configured:
                    database.insert_document(
                        document_id=document_id,
                        filename=url,
                        file_type="url",
                        source=url,
                        processing_status="completed",
                        upload_date=datetime.utcnow(),
                    )
                    records = [
                        {
                            "chunk_id": c.chunk_id,
                            "document_id": c.document_id,
                            "filename": c.filename,
                            "page_number": c.page_number,
                            "text": c.text,
                            "source": c.source,
                            "metadata": {"source": c.source, "char_count": c.char_count},
                        }
                        for c in chunks
                    ]
                    for i in range(0, len(records), DB_INSERT_BATCH):
                        try:
                            database.insert_chunks(records[i : i + DB_INSERT_BATCH])
                        except Exception as be:
                            logger.warning("[UPLOAD_URL] batch insert failed (non-fatal): %s", be)
            except Exception as db_exc:
                logger.warning("[UPLOAD_URL] Supabase insert skipped (non-fatal): %s", db_exc)

            job.update(
                status="completed",
                stage="Completed",
                progress_pct=100,
                chunks_done=total,
                chunks_total=total,
                finished_at=datetime.utcnow().isoformat(),
            )
        except ExtractionError as exc:
            err_msg = str(exc).lower()
            detail = (
                "No readable text was found at this URL."
                if ("readable text" in err_msg or "empty" in err_msg)
                else f"URL extraction failed: {exc}"
            )
            logger.error("[UPLOAD_URL] ExtractionError: %s", exc)
            job.update(status="failed", stage="Failed", error=detail, finished_at=datetime.utcnow().isoformat())
        except Exception as exc:
            logger.exception("[UPLOAD_URL] Unexpected error: %s", exc)
            job.update(
                status="failed",
                stage="Failed",
                error=f"{type(exc).__name__}: {exc}",
                finished_at=datetime.utcnow().isoformat(),
            )

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_url)

    return {
        "document_id": document_id,
        "job_id": job_id,
        "filename": url,
        "status": "indexing",
        "message": "URL ingestion started. Poll /api/upload/status/{job_id} for progress.",
    }
