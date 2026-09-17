import os
import json
from datetime import datetime
from typing import Dict, Any

# Locate backend/data directory
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
metadata_dir = os.path.join(backend_dir, "data")
metadata_file_path = os.path.join(metadata_dir, "metadata.json")

# Ensure metadata directory exists
os.makedirs(metadata_dir, exist_ok=True)

def load_all_metadata() -> Dict[str, Any]:
    if not os.path.exists(metadata_file_path):
        return {}
    try:
        with open(metadata_file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_all_metadata(data: Dict[str, Any]) -> None:
    try:
        with open(metadata_file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving metadata: {e}")

def save_document_metadata(document_id: str, metadata: Dict[str, Any]) -> None:
    all_meta = load_all_metadata()
    # Serialize datetime if any
    serialized_meta = {}
    for k, v in metadata.items():
        if isinstance(v, datetime):
            serialized_meta[k] = v.isoformat()
        else:
            serialized_meta[k] = v
    all_meta[document_id] = serialized_meta
    save_all_metadata(all_meta)

def get_document_metadata(document_id: str) -> Dict[str, Any]:
    all_meta = load_all_metadata()
    return all_meta.get(document_id, {})

def store_contradictions(document_id: str, contradictions: list[Dict[str, Any]], query: str = "") -> None:
    """Persist detected contradictions for a document so they can later be viewed in Evidence Viewer."""
    if not document_id or not contradictions:
        return
    all_meta = load_all_metadata()
    doc_meta = all_meta.get(document_id, {})
    existing = doc_meta.get("contradictions", [])
    for c in contradictions:
        entry = dict(c)
        if query and "query" not in entry:
            entry["query"] = query
        entry["timestamp"] = datetime.utcnow().isoformat()
        pair_key = (
            str(entry.get("chunk_a") or entry.get("chunk_id_a") or ""),
            str(entry.get("chunk_b") or entry.get("chunk_id_b") or ""),
        )
        is_dup = any(
            (
                str(e.get("chunk_a") or e.get("chunk_id_a") or ""),
                str(e.get("chunk_b") or e.get("chunk_id_b") or ""),
            ) == pair_key
            for e in existing
        )
        if not is_dup:
            existing.append(entry)
    doc_meta["contradictions"] = existing
    all_meta[document_id] = doc_meta
    save_all_metadata(all_meta)

def get_contradictions(document_id: str) -> list[Dict[str, Any]]:
    """Retrieve persisted contradictions for a document."""
    if not document_id:
        return []
    all_meta = load_all_metadata()
    return all_meta.get(document_id, {}).get("contradictions", [])

