"""Document upload and listing routes."""
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

# Ensure documents directory exists
DOCUMENTS_DIR = Path(settings.documents_path)
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)


# task_executor = ThreadPoolExecutor(max_workers=4)


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


class DocumentListResponse(BaseModel):
    """List of stored documents."""

    documents: List[StoredDocument]


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """Accept a PDF upload and store it for later use."""

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")

    if len(content) > settings.max_upload_size:
        max_mb = round(settings.max_upload_size / (1024 * 1024))
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size: {max_mb} MB")

    safe_name = Path(file.filename).name
    stored_filename = f"{uuid.uuid4().hex}_{safe_name}"
    file_path = DOCUMENTS_DIR / stored_filename
    file_path.write_bytes(content)

    uploaded_at = datetime.utcnow().isoformat() + "Z"
    url = f"/documents/{stored_filename}"

    logger.info("Stored PDF upload at %s", file_path)

    return DocumentUploadResponse(
        filename=safe_name,
        stored_filename=stored_filename,
        url=url,
        size_bytes=len(content),
        uploaded_at=uploaded_at,
        message="Document uploaded successfully",
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
