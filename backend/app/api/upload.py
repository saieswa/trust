import os
import uuid
import shutil
from datetime import datetime
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from app.database.metadata import save_document_metadata

router = APIRouter(prefix="/api")

SUPPORTED_EXTENSIONS = {"pdf", "txt", "doc", "docx", "csv", "json"}

# Base directory for the backend
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
uploads_dir = os.path.join(backend_dir, "data", "uploads")

# Ensure uploads directory exists
os.makedirs(uploads_dir, exist_ok=True)

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # 1. Validate file extension
    filename = file.filename
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided"
        )
        
    _, ext = os.path.splitext(filename)
    ext = ext.lower().lstrip(".")
    
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: .{ext}. Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
        
    # 2. Generate unique document_id
    document_id = str(uuid.uuid4())
    
    # 3. Define save path and save the file
    # Use <document_id>_<filename> to ensure it's completely unique and won't overwrite anything
    safe_filename = f"{document_id}_{filename}"
    file_path = os.path.join(uploads_dir, safe_filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}"
        )
        
    # 4. Store metadata
    metadata = {
        "document_id": document_id,
        "filename": filename,
        "file_type": ext,
        "upload_timestamp": datetime.utcnow(),
        "file_path": file_path,
        "processing_status": "uploaded"
    }
    
    try:
        save_document_metadata(document_id, metadata)
    except Exception as e:
        # If metadata save fails, clean up the saved file
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save document metadata: {str(e)}"
        )
        
    # 5. Return JSON response
    return {
        "document_id": document_id,
        "filename": filename,
        "status": "success"
    }
