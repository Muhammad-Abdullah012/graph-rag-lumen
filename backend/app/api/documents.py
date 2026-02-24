"""Document upload and listing routes."""
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import aiofiles
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from config.settings import settings

_CHUNK_SIZE = 1024 * 64  # 64 KB per read chunk

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

# Ensure documents directory exists
DOCUMENTS_DIR = Path(settings.documents_path)
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)


class StoredDocument(BaseModel):
    """Metadata for a stored PDF."""

    filename: str
    stored_filename: str
    url: str
    size_bytes: int
    uploaded_at: str


class DocumentUploadResponse(StoredDocument):
    """Upload response payload."""

    message: str
    processing_started: bool = False


class DocumentListResponse(BaseModel):
    """List of stored documents."""

    documents: List[StoredDocument]


class ProcessingStatusResponse(BaseModel):
    """Processing status for a document."""

    filename: str
    status: str
    step: str
    error: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """Accept a PDF upload, store it, and start OCR pipeline in background.

    The file is streamed directly to disk in 64 KB chunks — no full-file
    buffering in RAM — so large PDFs upload quickly without blocking the
    event loop.
    """

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    safe_name = Path(file.filename).name
    stored_filename = f"{uuid.uuid4().hex}_{safe_name}"
    file_path = DOCUMENTS_DIR / stored_filename

    total_bytes = 0
    max_bytes = settings.max_upload_size
    max_mb = round(max_bytes / (1024 * 1024))

    # Stream upload: write chunks directly to disk without loading into RAM
    async with aiofiles.open(file_path, "wb") as out:
        while True:
            chunk = await file.read(_CHUNK_SIZE)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                await out.close()
                file_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size: {max_mb} MB",
                )
            await out.write(chunk)

    if total_bytes == 0:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded file is empty")

    uploaded_at = datetime.utcnow().isoformat() + "Z"
    url = f"/documents/{stored_filename}"

    logger.info("Stored PDF upload at %s (%d bytes)", file_path, total_bytes)

    # Start OCR pipeline in background
    processing_started = False
    try:
        from backend.app.modules.ocr_pipeline import process_document_background
        process_document_background(str(file_path), stored_filename)
        processing_started = True
        logger.info("Started background OCR pipeline for %s", stored_filename)
    except Exception as e:
        logger.warning("Could not start OCR pipeline: %s", e)

    return DocumentUploadResponse(
        filename=safe_name,
        stored_filename=stored_filename,
        url=url,
        size_bytes=total_bytes,
        uploaded_at=uploaded_at,
        message="Document uploaded successfully"
        + (" — OCR processing started in background." if processing_started else ""),
        processing_started=processing_started,
    )


@router.get("/processing-status", response_model=List[ProcessingStatusResponse])
async def get_all_processing_status():
    """Return processing status for all documents."""
    from backend.app.modules.ocr_pipeline import get_all_processing_statuses
    rows = get_all_processing_statuses()  # list of dicts from Postgres
    result = []
    for row in rows:
        result.append(ProcessingStatusResponse(
            filename=row.get("filename", ""),
            status=row.get("status", "unknown"),
            step=row.get("step", "unknown"),
            error=row.get("error"),
            stats=row.get("stats"),
        ))
    return result


@router.get("/processing-status/{filename}", response_model=ProcessingStatusResponse)
async def get_document_processing_status(filename: str):
    """Return processing status for a specific document."""
    from backend.app.modules.ocr_pipeline import get_processing_status
    status = get_processing_status(filename)
    if not status:
        raise HTTPException(status_code=404, detail="No processing status found for this document")
    return ProcessingStatusResponse(
        filename=filename,
        status=status.get("status", "unknown"),
        step=status.get("step", "unknown"),
        error=status.get("error"),
        stats=status.get("stats"),
    )


@router.get("", response_model=DocumentListResponse)
@router.get("/", response_model=DocumentListResponse, include_in_schema=False)
async def list_documents() -> DocumentListResponse:
    """Return metadata for all stored PDFs, newest first."""

    documents: List[StoredDocument] = []

    pdf_files = [p for p in DOCUMENTS_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]

    for path in sorted(pdf_files, key=lambda p: p.stat().st_mtime, reverse=True):
        stat = path.stat()
        original_name = path.name.split("_", 1)[1] if "_" in path.name else path.name

        documents.append(
            StoredDocument(
                filename=original_name,
                stored_filename=path.name,
                url=f"/documents/{path.name}",
                size_bytes=stat.st_size,
                uploaded_at=datetime.utcfromtimestamp(stat.st_mtime).isoformat() + "Z",
            )
        )

    return DocumentListResponse(documents=documents)
