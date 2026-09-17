import logging
import os
import shutil
import uuid
from datetime import datetime

from fastapi import APIRouter, File, HTTPException, UploadFile, status
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
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB max upload size

# Base directory for the backend
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
uploads_dir = os.path.join(backend_dir, "data", "uploads")
_vector_store: FAISSStore | None = None

# Ensure uploads directory exists
os.makedirs(uploads_dir, exist_ok=True)


def get_vector_store() -> FAISSStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = FAISSStore()
    return _vector_store


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # 1. Validate file extension and sanitize filename
    filename = file.filename
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided",
        )

    # Sanitize filename to eliminate any directory traversal sequences
    clean_filename = os.path.basename(filename.replace("\\", "/"))
    if not clean_filename or clean_filename in (".", ".."):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename provided.",
        )

    _, ext = os.path.splitext(clean_filename)
    ext = ext.lower().lstrip(".")

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: .{ext}. Supported types: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    # 2. Generate unique document_id
    document_id = str(uuid.uuid4())

    # 3. Define save path and securely verify target is within uploads_dir
    safe_filename = f"{document_id}_{clean_filename}"
    file_path = os.path.abspath(os.path.join(uploads_dir, safe_filename))
    uploads_dir_abs = os.path.abspath(uploads_dir)
    if not file_path.startswith(uploads_dir_abs):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename: target destination is outside upload directory.",
        )

    try:
        os.makedirs(uploads_dir, exist_ok=True)
        total_bytes = 0
        chunk_size = 1024 * 1024  # 1MB buffer
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE_BYTES:
                    buffer.close()
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB.",
                    )
                buffer.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to save uploaded file to disk: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document could not be processed.",
        )

    logger.info("[UPLOAD] filename=%s", clean_filename)
    logger.info("[UPLOAD] document_id=%s", document_id)

    # 4. Processing pipeline: extraction -> chunking -> embedding/FAISS -> metadata save
    try:
        database = get_database()
        if database.configured:
            database.insert_document(
                document_id=document_id,
                filename=filename,
                file_type=ext,
                source=file_path,
                processing_status="uploaded",
                upload_date=datetime.utcnow(),
            )

        extracted = load_document(
            file_path,
            document_id=document_id,
            source_filename=filename,
        )
        total_chars = sum(len(p.text) for p in extracted.pages)
        logger.info("[INGESTION] extracted characters=%d", total_chars)

        chunks = chunk_document(extracted)
        logger.info("[CHUNKING] chunks=%d", len(chunks))

        store = get_vector_store()
        vectors_before = store.size
        logger.info(
            "[EMBEDDING] model=all-MiniLM-L6-v2 dimension=%d vectors=%d",
            store.dimension,
            len(chunks),
        )
        logger.info(
            "[FAISS] index dimension=%d vectors before=%d",
            store.dimension,
            vectors_before,
        )

        store.add_chunks(chunks)
        logger.info("[FAISS] vectors after=%d", store.size)

        if database.configured:
            logger.info("[DATABASE] saving metadata")
            database.insert_chunks(
                [
                    {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "filename": chunk.filename,
                        "page_number": chunk.page_number,
                        "text": chunk.text,
                        "source": chunk.source,
                        "metadata": {
                            "source": chunk.source,
                            "char_count": chunk.char_count,
                            "word_count": chunk.word_count,
                        },
                    }
                    for chunk in chunks
                ]
            )

        try:
            save_document_metadata(
                document_id,
                {
                    "document_id": document_id,
                    "filename": filename,
                    "file_type": ext,
                    "upload_date": datetime.utcnow().isoformat(),
                    "chunks_count": len(chunks),
                },
            )
        except Exception as save_err:
            logger.warning("Failed saving local document metadata: %s", save_err)

        logger.info("[UPLOAD] completed successfully")
    except ExtractionError as exc:
        logger.error("[EXTRACTION ERROR] %s", exc)
        if os.path.exists(file_path):
            os.remove(file_path)
        err_msg = str(exc).lower()
        detail = "No readable text was found in this document." if ("readable text" in err_msg or "empty" in err_msg) else "Document could not be processed."
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
        ) from exc
    except (ChunkingError, EmbeddingError, FAISSStoreError, SupabaseDatabaseError) as exc:
        logger.exception("[PROCESSING ERROR] %s: %s", type(exc).__name__, exc)
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document could not be processed.",
        ) from exc
    except Exception as e:
        logger.exception("[UPLOAD ERROR] Unexpected error during upload pipeline: %s", e)
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document could not be processed.",
        ) from e

    # 5. Return JSON response
    return {
        "document_id": document_id,
        "filename": clean_filename,
        "status": "success",
    }


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
    try:
        extracted = load_document(url, document_id=document_id)
        chunks = chunk_document(extracted)
        store = get_vector_store()
        store.add_chunks(chunks)

        database = get_database()
        if database.configured:
            database.insert_document(
                document_id=document_id,
                filename=url,
                file_type="url",
                source=url,
                processing_status="uploaded",
                upload_date=datetime.utcnow(),
            )
            database.insert_chunks([
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "filename": chunk.filename,
                    "page_number": chunk.page_number,
                    "text": chunk.text,
                    "source": chunk.source,
                    "metadata": {"source": chunk.source, "char_count": chunk.char_count},
                }
                for chunk in chunks
            ])

        try:
            save_document_metadata(
                document_id,
                {
                    "document_id": document_id,
                    "filename": url,
                    "file_type": "url",
                    "upload_date": datetime.utcnow().isoformat(),
                    "chunks_count": len(chunks),
                },
            )
        except Exception as save_err:
            logger.warning("Failed saving local document metadata for URL: %s", save_err)

    except ExtractionError as exc:
        logger.error("[EXTRACTION ERROR URL] %s", exc)
        err_msg = str(exc).lower()
        detail = "No readable text was found in this document." if ("readable text" in err_msg or "empty" in err_msg) else "Document could not be processed."
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
        ) from exc
    except Exception as exc:
        logger.exception("URL ingestion failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document could not be processed.",
        ) from exc

    return {
        "document_id": document_id,
        "filename": url,
        "status": "success",
    }

