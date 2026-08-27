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
