"""Graph Management Routes - Build & inspect the knowledge graph"""
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

from backend.app.modules.graph_builder import get_graph_builder
from backend.app.modules.graph_querier import get_graph_querier

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/graph", tags=["graph"])


class GraphBuildResponse(BaseModel):
    """Response after building the graph"""
    status: str
    stats: Dict[str, Any]


class GraphStatsResponse(BaseModel):
    """Graph statistics response"""
    stats: Dict[str, Any]


@router.post("/build", response_model=GraphBuildResponse)
async def build_graph() -> GraphBuildResponse:
    """
    Build the knowledge graph from all JSON files in backend/json/.
    This is idempotent (uses MERGE) so can be called multiple times safely.
    """
    try:
        builder = get_graph_builder()
        stats = builder.build_all()
        logger.info(f"Graph built: {stats}")
        return GraphBuildResponse(status="success", stats=stats)
    except Exception as e:
        logger.error(f"Graph build failed: {str(e)}")
        return GraphBuildResponse(status=f"error: {str(e)}", stats={})


@router.post("/rebuild", response_model=GraphBuildResponse)
async def rebuild_graph() -> GraphBuildResponse:
    """
    Clear the graph and rebuild from scratch.
    """
    try:
        builder = get_graph_builder()
        builder.clear_graph()
        stats = builder.build_all()
        logger.info(f"Graph rebuilt: {stats}")
        return GraphBuildResponse(status="success", stats=stats)
    except Exception as e:
        logger.error(f"Graph rebuild failed: {str(e)}")
        return GraphBuildResponse(status=f"error: {str(e)}", stats={})


@router.get("/stats", response_model=GraphStatsResponse)
async def graph_stats() -> GraphStatsResponse:
    """
    Get current graph node counts.
    """
    try:
        querier = get_graph_querier()
        stats = querier.get_graph_stats()
        return GraphStatsResponse(stats=stats)
    except Exception as e:
        logger.error(f"Could not get graph stats: {str(e)}")
        return GraphStatsResponse(stats={"error": str(e)})


@router.post("/process-document/{filename}")
async def process_document_manually(filename: str):
    """
    Manually trigger OCR pipeline for a document already in the documents folder.
    Useful for re-processing or processing documents uploaded before the pipeline existed.
    """
    from config.settings import settings
    docs_dir = Path(settings.documents_path)

    # Find the file
    file_path = docs_dir / filename
    if not file_path.exists():
        # Try to find by partial match
        matches = [p for p in docs_dir.iterdir() if filename in p.name and p.suffix.lower() == ".pdf"]
        if not matches:
            raise HTTPException(status_code=404, detail=f"Document not found: {filename}")
        file_path = matches[0]

    try:
        from backend.app.modules.ocr_pipeline import process_document_background
        process_document_background(str(file_path), file_path.name)
        return {"status": "processing_started", "filename": file_path.name}
    except Exception as e:
        logger.error(f"Failed to start processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-all-documents")
async def process_all_documents():
    """
    Trigger OCR pipeline for all PDF documents in the documents folder.
    Each document is processed in a background thread.
    """
    from config.settings import settings
    docs_dir = Path(settings.documents_path)

    pdf_files = [p for p in docs_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]

    if not pdf_files:
        return {"status": "no_documents", "message": "No PDF files found in documents folder."}

    started = []
    try:
        from backend.app.modules.ocr_pipeline import process_document_background
        for pdf_path in pdf_files:
            process_document_background(str(pdf_path), pdf_path.name)
            started.append(pdf_path.name)
    except Exception as e:
        logger.error(f"Failed to start batch processing: {e}")

    return {
        "status": "processing_started",
        "documents": started,
        "count": len(started),
    }


class SemanticSearchResponse(BaseModel):
    """Semantic search results"""
    query: str
    results: list


@router.post("/semantic-search", response_model=SemanticSearchResponse)
async def semantic_search_endpoint(query: str, top_k: int = 10):
    """
    Perform semantic search across all document content.
    """
    try:
        querier = get_graph_querier()
        results = querier.semantic_search(query, top_k=top_k)
        return SemanticSearchResponse(query=query, results=results)
    except Exception as e:
        logger.error(f"Semantic search failed: {str(e)}")
        return SemanticSearchResponse(query=query, results=[])
