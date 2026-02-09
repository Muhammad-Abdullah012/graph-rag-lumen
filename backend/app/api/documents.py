"""Document Processing Routes"""
import logging
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel

from backend.app.modules.extraction import get_pdf_extractor
from backend.app.modules.splitting import TextSplitter
from backend.app.modules.graph_builder import GraphBuilder
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

# Ensure documents directory exists
Path(settings.documents_path).mkdir(parents=True, exist_ok=True)


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


# Store processing status in memory (in production, use database)
processing_status = {}


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> DocumentUploadResponse:
    """
    Upload and process PDF document
    
    Args:
        file: PDF file
        background_tasks: Background task queue
        
    Returns:
        Upload response with document ID
    """
    try:
        # Validate file
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        if file.size > settings.max_upload_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.max_upload_size} bytes"
            )
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Save file
        file_path = Path(settings.documents_path) / f"{document_id}_{file.filename}"
        with open(file_path, 'wb') as f:
            f.write(await file.read())
        
        logger.info(f"Saved uploaded file: {file_path}")
        
        # Create relative URL for document
        document_url = f"documents/{document_id}_{file.filename}"
        
        # Queue background task to process document
        background_tasks.add_task(
            process_document,
            document_id=document_id,
            file_path=str(file_path),
            filename=file.filename,
            document_url=document_url
        )
        
        processing_status[document_id] = {
            "status": "pending",
            "message": "Document queued for processing"
        }
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            message="Document uploaded successfully and queued for processing",
            status="pending"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_document(
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
            "message": "Extracting text from PDF..."
        }
        
        logger.info(f"Processing document {document_id}: {filename}")
        
        # Extract text from PDF
        extractor = get_pdf_extractor()
        extracted_text = extractor.extract(file_path)
        
        processing_status[document_id]["message"] = "Splitting text into chunks..."
        
        # Split text
        splitter = TextSplitter()
        chunks = splitter.split(extracted_text)
        
        processing_status[document_id]["message"] = "Building knowledge graph..."
        
        # Build graph
        builder = GraphBuilder()
        builder.build_graph(
            chunks=chunks,
            document_id=document_id,
            document_name=filename,
            document_url=document_url
        )
        
        processing_status[document_id] = {
            "status": "completed",
            "message": f"Successfully processed {len(chunks)} chunks"
        }
        
        logger.info(f"Successfully processed document {document_id}")
        
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}")
        processing_status[document_id] = {
            "status": "failed",
            "message": f"Error: {str(e)}"
        }


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
