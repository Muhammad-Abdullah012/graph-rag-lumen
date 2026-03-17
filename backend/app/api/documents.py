"""Document Processing Routes"""
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import aiofiles
import asyncio

from backend.app.modules.ocr_pipeline import get_ocr_pipeline
from backend.app.modules.graph_builder import GraphBuilder
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

# Ensure documents directory exists
Path(settings.documents_path).mkdir(parents=True, exist_ok=True)

task_executor = ThreadPoolExecutor(max_workers=4)


class DocumentUploadResponse(BaseModel):
    """Response for document upload"""
    document_id: str
    filename: str
    message: str
    status: str


class ProcessingStatus(BaseModel):
    """Document processing status"""
    document_id: str
    status: str  # pending, processing, completed, failed
    message: str


class DocumentInfo(BaseModel):
    """Uploaded document file info"""
    document_id: str
    stored_filename: str
    filename: str
    size_bytes: int
    uploaded_at: str
    url: str


# Store processing status in memory (in production, use database)
processing_status = {}


@router.get("")
async def list_documents():
    """List all uploaded PDF files with their metadata."""
    docs_dir = Path(settings.documents_path)
    loop = asyncio.get_running_loop()

    def _scan():
        files = [f for f in docs_dir.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"]
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        result = []
        for pdf_file in files:
            stat = pdf_file.stat()
            parts = pdf_file.name.split("_", 1)
            document_id = parts[0]
            filename = parts[1] if len(parts) == 2 else pdf_file.name
            result.append(DocumentInfo(
                document_id=document_id,
                stored_filename=pdf_file.name,
                filename=filename,
                size_bytes=stat.st_size,
                uploaded_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                url=f"{settings.documents_base_url}/{pdf_file.name}",
            ))
        return result

    result = await loop.run_in_executor(task_executor, _scan)
    return {"documents": result}


@router.post("/upload", response_model=List[DocumentUploadResponse])
async def upload_documents(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> List[DocumentUploadResponse]:
    """Upload and process one or more PDF documents."""
    responses = []
    for file in files:
        try:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"{file.filename}: only PDF files are allowed")

            if file.size > settings.max_upload_size:
                raise HTTPException(
                    status_code=413,
                    detail=f"{file.filename}: file too large (max {settings.max_upload_size} bytes)"
                )

            document_id = str(uuid.uuid4())
            file_path = Path(settings.documents_path) / f"{document_id}_{file.filename}"

            async with aiofiles.open(file_path, 'wb') as f:
                while chunk := await file.read(settings.upload_chunk_size):
                    await f.write(chunk)

            logger.info(f"Saved uploaded file: {file_path}")

            document_url = f"documents/{document_id}_{file.filename}"

            background_tasks.add_task(
                process_document,
                document_id=document_id,
                file_path=str(file_path),
                filename=file.filename,
                document_url=document_url,
            )

            processing_status[document_id] = {
                "status": "pending",
                "message": "Document queued for processing",
            }

            responses.append(DocumentUploadResponse(
                document_id=document_id,
                filename=file.filename,
                message="Uploaded successfully and queued for processing",
                status="pending",
            ))

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error uploading {file.filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    return responses


def _process_document_sync(
    document_id: str,
    file_path: str,
    filename: str,
    document_url: str
):
    """
    Background task to process document
    
    Args:
        document_id: Unique document ID
        file_path: Path to saved PDF file
        filename: Original filename
        document_url: Relative URL to document
    """
    try:
        processing_status[document_id] = {
            "status": "processing",
            "message": "Running OCR on PDF..."
        }

        logger.info(f"Processing document {document_id}: {filename}")

        ocr = get_ocr_pipeline()
        pages = ocr.load_existing(filename)
        if pages is None:
            pages = ocr.process(
                pdf_path=file_path,
                document_id=document_id,
                filename=filename,
            )
        else:
            logger.info(f"Skipping OCR for {filename}: using cached JSON")

        processing_status[document_id]["message"] = "Building knowledge graph..."

        # Build graph (Document + Page nodes)
        builder = GraphBuilder()
        builder.build_graph(
            pages=pages,
            document_id=document_id,
            document_name=filename,
            document_url=document_url,
        )

        processing_status[document_id] = {
            "status": "completed",
            "message": f"Successfully processed {len(pages)} pages"
        }
        
        logger.info(f"Successfully processed document {document_id}")
        
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}")
        processing_status[document_id] = {
            "status": "failed",
            "message": f"Error: {str(e)}"
        }

async def process_document(document_id, file_path, filename, document_url):
    """Async wrapper that offloads to thread pool"""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        task_executor,
        _process_document_sync,
        document_id,
        file_path,
        filename,
        document_url
    )

@router.post("/process-all")
async def process_all_documents(background_tasks: BackgroundTasks = BackgroundTasks()):
    """Queue all PDFs in the documents directory for OCR processing."""
    docs_dir = Path(settings.documents_path)
    pdf_files = list(docs_dir.glob("*.pdf"))

    if not pdf_files:
        return {"message": "No PDF files found", "queued": 0}

    queued = []
    for pdf_file in pdf_files:
        parts = pdf_file.name.split("_", 1)
        document_id = parts[0]
        filename = parts[1] if len(parts) == 2 else pdf_file.name

        if processing_status.get(document_id, {}).get("status") == "processing":
            continue

        processing_status[document_id] = {"status": "pending", "message": "Queued for processing"}
        background_tasks.add_task(
            process_document,
            document_id=document_id,
            file_path=str(pdf_file),
            filename=pdf_file.name,
            document_url=f"documents/{pdf_file.name}",
        )
        queued.append(filename)

    return {"message": "Processing started", "queued": len(queued), "files": queued}


@router.get("/status/{document_id}", response_model=ProcessingStatus)
async def get_processing_status(document_id: str) -> ProcessingStatus:
    """
    Get processing status of document
    
    Args:
        document_id: Document ID
        
    Returns:
        Processing status
    """
    if document_id not in processing_status:
        raise HTTPException(status_code=404, detail="Document not found")
    
    status_info = processing_status[document_id]
    return ProcessingStatus(
        document_id=document_id,
        status=status_info.get("status", "unknown"),
        message=status_info.get("message", "")
    )
