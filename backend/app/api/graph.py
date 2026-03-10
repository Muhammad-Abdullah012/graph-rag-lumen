"""Graph Management Routes - Build & inspect the Graph-RAG knowledge graph"""
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

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
    Creates the full Graph-RAG schema: Document → Chapter → Page → Section,
    plus Tables, Figures, Formulas, Concepts with MENTIONS & RELATED_TO edges.
    Idempotent (uses MERGE).
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
    """Clear the graph and rebuild from scratch."""
    try:
        builder = get_graph_builder()
        builder.clear_graph()
        stats = builder.build_all()
        logger.info(f"Graph rebuilt: {stats}")
        return GraphBuildResponse(status="success", stats=stats)
    except Exception as e:
        logger.error(f"Graph rebuild failed: {str(e)}")
        return GraphBuildResponse(status=f"error: {str(e)}", stats={})


@router.post("/generate-embeddings")
async def generate_embeddings():
    """Generate embeddings for all Section and Concept nodes that lack them."""
    try:
        builder = get_graph_builder()
        count = builder.generate_embeddings()
        return {"status": "success", "embeddings_created": count}
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        return {"status": f"error: {str(e)}", "embeddings_created": 0}


@router.post("/generate-summaries")
async def generate_summaries():
    """Generate rich chapter and document summaries for top-down retrieval.

    Must run after graph build (sections, formulas, tables exist) and
    before embedding generation (so summaries can be embedded).
    """
    try:
        builder = get_graph_builder()
        count = builder.generate_summaries()
        return {"status": "success", "summaries_created": count}
    except Exception as e:
        logger.error(f"Summary generation failed: {str(e)}")
        return {"status": f"error: {str(e)}", "summaries_created": 0}


@router.post("/compute-similarity")
async def compute_similarity(threshold: float = 0.80, top_k: int = 5):
    """Compute SEMANTICALLY_SIMILAR edges between sections across documents."""
    try:
        builder = get_graph_builder()
        count = builder.compute_semantic_similarity(threshold=threshold, top_k=top_k)
        return {"status": "success", "similarity_links_created": count}
    except Exception as e:
        logger.error(f"Similarity computation failed: {str(e)}")
        return {"status": f"error: {str(e)}", "similarity_links_created": 0}


@router.get("/stats", response_model=GraphStatsResponse)
async def graph_stats() -> GraphStatsResponse:
    """Get current graph node and relationship counts."""
    try:
        querier = get_graph_querier()
        stats = querier.get_graph_stats()
        return GraphStatsResponse(stats=stats)
    except Exception as e:
        logger.error(f"Could not get graph stats: {str(e)}")
        return GraphStatsResponse(stats={"error": str(e)})


@router.get("/documents")
async def list_documents():
    """List all documents in the knowledge graph."""
    try:
        querier = get_graph_querier()
        return {"documents": querier.list_documents()}
    except Exception as e:
        return {"error": str(e)}


@router.get("/chapters")
async def list_chapters(document: Optional[str] = None):
    """List chapters, optionally filtered by document keyword."""
    try:
        querier = get_graph_querier()
        return {"chapters": querier.list_chapters(document)}
    except Exception as e:
        return {"error": str(e)}


@router.get("/concepts")
async def search_concepts(q: str, limit: int = 15):
    """Search concepts by keyword."""
    try:
        querier = get_graph_querier()
        return {"concepts": querier.search_concepts(q, limit=limit)}
    except Exception as e:
        return {"error": str(e)}


@router.post("/process-document/{filename}")
async def process_document_manually(filename: str):
    """
    Manually trigger OCR pipeline for a document already in the documents folder.
    """
    from config.settings import settings
    docs_dir = Path(settings.documents_path)

    file_path = docs_dir / filename
    if not file_path.exists():
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
    """Trigger OCR pipeline for all PDFs in the documents folder."""
    from config.settings import settings
    docs_dir = Path(settings.documents_path)

    pdf_files = [p for p in docs_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
    if not pdf_files:
        return {"status": "no_documents", "message": "No PDF files found."}

    started = []
    try:
        from backend.app.modules.ocr_pipeline import process_document_background
        for pdf_path in pdf_files:
            process_document_background(str(pdf_path), pdf_path.name)
            started.append(pdf_path.name)
    except Exception as e:
        logger.error(f"Failed to start batch processing: {e}")

    return {"status": "processing_started", "documents": started, "count": len(started)}


class SemanticSearchResponse(BaseModel):
    """Semantic search results"""
    query: str
    results: list


@router.post("/semantic-search", response_model=SemanticSearchResponse)
async def semantic_search_endpoint(query: str, top_k: int = 10):
    """Perform semantic vector search across all Section embeddings."""
    try:
        querier = get_graph_querier()
        results = querier.semantic_search(query, top_k=top_k)
        return SemanticSearchResponse(query=query, results=results)
    except Exception as e:
        logger.error(f"Semantic search failed: {str(e)}")
        return SemanticSearchResponse(query=query, results=[])
