import logging
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.chat import router as chat_router
from app.api.upload import router as upload_router
from app.api.critic import router as critic_router
from app.api.compare import router as compare_router
from app.api.evaluation_api import router as evaluation_router
from app.api.dashboard import router as dashboard_router
from app.ingestion.loaders import ExtractionError
from app.llm.groq_client import GroqClientError

logger = logging.getLogger(__name__)

app = FastAPI(title="Trust-Aware Multi-Agent RAG Framework")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(chat_router)
app.include_router(upload_router)
app.include_router(critic_router)
app.include_router(compare_router)
app.include_router(evaluation_router)
app.include_router(dashboard_router)


@app.on_event("startup")
async def startup_warmup():
    """Preload in-memory models (Embedding model, FAISS store, XGBoost) to eliminate first-query cold start."""
    logger.info("[PERF] Starting background model pre-warming...")
    try:
        from app.embeddings.embedding_model import get_embedding_model
        _ = get_embedding_model().dimension
        logger.info("[PERF] Embedding model preloaded.")
    except Exception as exc:
        logger.warning("[PERF] Embedding model warmup note: %s", exc)

    try:
        from app.api.chat import get_retriever
        retriever = get_retriever()
        if hasattr(retriever.store, "reload_if_modified"):
            retriever.store.reload_if_modified()
        elif retriever.store.index is None:
            retriever.store.load()
        logger.info("[PERF] FAISS vector store preloaded.")
    except Exception as exc:
        logger.warning("[PERF] FAISS store warmup note: %s", exc)

    try:
        from app.trust.trust_model import TrustFeatures, predict_xgboost_trust_score
        _ = predict_xgboost_trust_score(
            TrustFeatures(
                relevance=1.0,
                evidence_support=1.0,
                source_quality=1.0,
                evidence_agreement=1.0,
                contradiction_ratio=0.0,
                retrieval_confidence=1.0,
                verifier_agreement=1.0,
            )
        )
        logger.info("[PERF] XGBoost model preloaded.")
    except Exception as exc:
        logger.warning("[PERF] XGBoost warmup note: %s", exc)


# ==============================================================================
# Global Security & User-Friendly Exception Handlers
# Prevents stack trace exposure and internal implementation leaks to clients.
# ==============================================================================

@app.exception_handler(GroqClientError)
async def groq_client_exception_handler(request: Request, exc: GroqClientError):
    logger.exception("Groq LLM Client Error on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Unable to connect to the language model."},
    )


@app.exception_handler(ExtractionError)
async def extraction_exception_handler(request: Request, exc: ExtractionError):
    logger.error("Document Extraction Error on %s %s: %s", request.method, request.url.path, exc)
    err_str = str(exc).lower()
    detail = "No readable text was found in this document." if ("readable text" in err_str or "empty" in err_str) else "Document could not be processed."
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": detail},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # Preserves clean HTTP status codes and details without leaking stack traces
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=exc.headers,
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # Full traceback is recorded strictly in server logs; clean message returned to client
    logger.exception("Unhandled application error on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Document could not be processed."},
    )


@app.get("/health")
def health():
    return {"status": "ok", "system": "Trust-Aware Multi-Agent RAG Framework"}