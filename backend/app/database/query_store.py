"""Query execution store and analytics engine for Trust-Aware RAG.

Records actual query executions, timings, trust scores, verification results,
and computes real aggregate system metrics without synthetic or hardcoded values.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# Base directories
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA_DIR = os.path.join(_BACKEND_DIR, "data")
_QUERIES_FILE = os.path.join(_DATA_DIR, "query_history.json")
_UPLOADS_DIR = os.path.join(_DATA_DIR, "uploads")
_METADATA_FILE = os.path.join(_DATA_DIR, "metadata.json")

os.makedirs(_DATA_DIR, exist_ok=True)


def _load_json(file_path: str, default_val: Any) -> Any:
    if not os.path.exists(file_path):
        return default_val
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Error reading %s: %s", file_path, exc)
        return default_val


def _save_json(file_path: str, data: Any) -> None:
    try:
        tmp_path = f"{file_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, file_path)
    except Exception as exc:
        logger.error("Failed saving to %s: %s", file_path, exc)


def load_query_history() -> list[dict[str, Any]]:
    """Load all recorded query executions."""
    data = _load_json(_QUERIES_FILE, [])
    return data if isinstance(data, list) else []


def record_query(
    question: str,
    document_id: str,
    trust_score: float | None,
    trust_level: str,
    verification_status: str,
    is_abstention: bool,
    contradictions_count: int,
    average_relevance: float | None,
    response_time_ms: float,
    answer: str = "",
    sources: list[Any] | None = None,
    evidence: list[Any] | None = None,
    contradictions: list[Any] | None = None,
) -> dict[str, Any]:
    """Persist an actual query execution event."""
    history = load_query_history()
    now_iso = datetime.now(timezone.utc).isoformat()
    record = {
        "query_id": str(uuid.uuid4()),
        "timestamp": now_iso,
        "document_id": document_id,
        "question": question,
        "trust_score": round(float(trust_score), 4) if trust_score is not None else None,
        "trust_level": str(trust_level).upper(),
        "verification_status": str(verification_status).upper(),
        "is_abstention": bool(is_abstention),
        "contradictions_count": int(contradictions_count),
        "average_relevance": round(float(average_relevance), 4) if average_relevance is not None else None,
        "response_time_ms": round(float(response_time_ms), 2),
        "answer": answer,
        "sources_count": len(sources or []),
        "evidence_count": len(evidence or []),
        "contradictions": contradictions or [],
    }
    history.append(record)
    _save_json(_QUERIES_FILE, history)
    return record


def get_actual_documents() -> list[dict[str, Any]]:
    """Return all actual stored documents across metadata.json and uploads directory."""
    docs_map: dict[str, dict[str, Any]] = {}

    # 1. From metadata.json
    metadata = _load_json(_METADATA_FILE, {})
    if isinstance(metadata, dict):
        for doc_id, info in metadata.items():
            if isinstance(info, dict):
                docs_map[doc_id] = {
                    "document_id": doc_id,
                    "filename": info.get("filename") or info.get("source_filename") or doc_id,
                    "upload_date": info.get("upload_date") or info.get("created_at"),
                }

    # 2. From uploads folder
    if os.path.exists(_UPLOADS_DIR):
        for fname in os.listdir(_UPLOADS_DIR):
            if fname == ".gitkeep" or not os.path.isfile(os.path.join(_UPLOADS_DIR, fname)):
                continue
            # Filenames are formatted as {document_id}_{original_filename}
            if "_" in fname:
                doc_id, original_name = fname.split("_", 1)
                if doc_id not in docs_map:
                    docs_map[doc_id] = {
                        "document_id": doc_id,
                        "filename": original_name,
                        "upload_date": None,
                    }
                elif not docs_map[doc_id].get("filename") or docs_map[doc_id]["filename"] == doc_id:
                    docs_map[doc_id]["filename"] = original_name

    # 3. Also include document_ids found in query history
    for q in load_query_history():
        did = q.get("document_id")
        if did and did not in docs_map:
            docs_map[did] = {
                "document_id": did,
                "filename": did,
                "upload_date": q.get("timestamp"),
            }

    return list(docs_map.values())


def get_system_metrics(
    document_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Compute aggregate dashboard metrics strictly from stored system data."""
    all_docs = get_actual_documents()
    all_queries = load_query_history()

    # Filter queries
    filtered_queries = all_queries

    if document_id and document_id.strip() and document_id != "all":
        filtered_queries = [q for q in filtered_queries if q.get("document_id") == document_id]

    if start_date:
        filtered_queries = [q for q in filtered_queries if q.get("timestamp", "") >= start_date]

    if end_date:
        filtered_queries = [q for q in filtered_queries if q.get("timestamp", "") <= end_date]

    total_questions = len(filtered_queries)
    total_documents = len(all_docs) if not document_id or document_id == "all" else (1 if any(d["document_id"] == document_id for d in all_docs) else 0)

    if total_questions == 0:
        return {
            "has_data": False,
            "total_documents": total_documents,
            "total_questions": 0,
            "average_trust_score": None,
            "high_trust_responses": 0,
            "medium_trust_responses": 0,
            "low_trust_abstained_responses": 0,
            "verified_answers": 0,
            "unsupported_answers": 0,
            "contradictions_detected": 0,
            "average_retrieval_relevance": None,
            "average_response_time_ms": None,
            "documents": all_docs,
            "queries": [],
            "trust_distribution": [],
            "verification_distribution": [],
            "timeline": [],
        }

    # 3. Average trust score
    scores = [q["trust_score"] for q in filtered_queries if q.get("trust_score") is not None]
    avg_trust_score = round(sum(scores) / len(scores), 4) if scores else None

    # 4. High-trust responses (score >= 0.75 or level HIGH)
    high_trust_count = sum(
        1 for q in filtered_queries
        if (q.get("trust_score") is not None and q["trust_score"] >= 0.75) or q.get("trust_level") == "HIGH"
    )

    # 5. Medium-trust responses (0.50 <= score < 0.75 or level MEDIUM)
    med_trust_count = sum(
        1 for q in filtered_queries
        if ((q.get("trust_score") is not None and 0.50 <= q["trust_score"] < 0.75) or q.get("trust_level") in ("MEDIUM", "MED"))
        and not q.get("is_abstention")
    )

    # 6. Low-trust / abstained responses
    low_trust_count = sum(
        1 for q in filtered_queries
        if (q.get("trust_score") is not None and q["trust_score"] < 0.50)
        or q.get("trust_level") == "LOW"
        or q.get("is_abstention") is True
    )

    # 7. Verified answers (status SUPPORTED)
    verified_count = sum(
        1 for q in filtered_queries
        if q.get("verification_status") in ("SUPPORTED", "VERIFIED") and not q.get("is_abstention")
    )

    # 8. Unsupported answers (status UNSUPPORTED or FAILED)
    unsupported_count = sum(
        1 for q in filtered_queries
        if q.get("verification_status") in ("UNSUPPORTED", "FAILED")
    )

    # 9. Contradictions detected
    contradictions_count = sum(q.get("contradictions_count", 0) for q in filtered_queries)

    # Also check stored contradictions in metadata.json for this document
    if contradictions_count == 0 and document_id and document_id != "all":
        meta = _load_json(_METADATA_FILE, {})
        doc_contras = meta.get(document_id, {}).get("contradictions", [])
        contradictions_count = len(doc_contras)
    elif contradictions_count == 0:
        meta = _load_json(_METADATA_FILE, {})
        total_meta_contras = sum(len(d.get("contradictions", [])) for d in meta.values() if isinstance(d, dict))
        contradictions_count = total_meta_contras

    # 10. Average retrieval relevance
    relevance_scores = [q["average_relevance"] for q in filtered_queries if q.get("average_relevance") is not None]
    avg_relevance = round(sum(relevance_scores) / len(relevance_scores), 4) if relevance_scores else None

    # 11. Average response time
    latencies = [q["response_time_ms"] for q in filtered_queries if q.get("response_time_ms") is not None]
    avg_response_time = round(sum(latencies) / len(latencies), 2) if latencies else None

    # Simple chart distributions
    trust_distribution = [
        {"name": "High Trust (≥75%)", "count": high_trust_count, "level": "HIGH"},
        {"name": "Medium Trust (50-74%)", "count": med_trust_count, "level": "MEDIUM"},
        {"name": "Low Trust / Abstained (<50%)", "count": low_trust_count, "level": "LOW"},
    ]

    verification_distribution = [
        {"name": "Verified (Supported)", "count": verified_count, "status": "SUPPORTED"},
        {"name": "Partially Supported", "count": sum(1 for q in filtered_queries if q.get("verification_status") == "PARTIALLY_SUPPORTED"), "status": "PARTIALLY_SUPPORTED"},
        {"name": "Unsupported / Failed", "count": unsupported_count, "status": "UNSUPPORTED"},
        {"name": "Abstained Refusals", "count": sum(1 for q in filtered_queries if q.get("is_abstention")), "status": "ABSTAINED"},
    ]

    # Activity timeline grouped by date (YYYY-MM-DD)
    timeline_map: dict[str, int] = {}
    for q in filtered_queries:
        ts = q.get("timestamp")
        if ts:
            day = ts.split("T")[0]
            timeline_map[day] = timeline_map.get(day, 0) + 1

    timeline = [{"date": day, "count": count} for day, count in sorted(timeline_map.items())]

    # Queries sorted latest first
    sorted_queries = sorted(filtered_queries, key=lambda x: x.get("timestamp", ""), reverse=True)

    return {
        "has_data": True,
        "total_documents": total_documents,
        "total_questions": total_questions,
        "average_trust_score": avg_trust_score,
        "high_trust_responses": high_trust_count,
        "medium_trust_responses": med_trust_count,
        "low_trust_abstained_responses": low_trust_count,
        "verified_answers": verified_count,
        "unsupported_answers": unsupported_count,
        "contradictions_detected": contradictions_count,
        "average_retrieval_relevance": avg_relevance,
        "average_response_time_ms": avg_response_time,
        "documents": all_docs,
        "queries": sorted_queries[:50],  # Latest 50 queries
        "trust_distribution": trust_distribution,
        "verification_distribution": verification_distribution,
        "timeline": timeline,
    }
