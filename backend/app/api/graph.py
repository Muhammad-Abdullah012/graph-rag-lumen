"""Graph Management Routes - Build & inspect the Graph-RAG knowledge graph"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.modules.graph_builder import get_graph_builder
from backend.app.modules.graph_jobs_db import (
    fail_job,
    finish_job,
    get_job_history,
    get_latest_job,
    get_running_job,
    start_job,
)
from backend.app.modules.graph_querier import get_graph_querier

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/graph", tags=["graph"])

# One worker so only one heavy job runs at a time.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="graph_worker")


# ── Models ────────────────────────────────────────────────────────────────────

class GraphBuildResponse(BaseModel):
    status: str
    stats: Dict[str, Any]


class GraphStatsResponse(BaseModel):
    stats: Dict[str, Any]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _assert_idle() -> None:
    """Raise 409 if a graph job is already running."""
    running = get_running_job()
    if running:
        raise HTTPException(
            status_code=409,
            detail=f"Job '{running['job_name']}' is already running (id={running['id']}). "
                   "Poll GET /api/graph/job-status for progress.",
        )


async def _dispatch(name: str, fn: Callable, *args) -> Dict[str, Any]:
    """Persist a job row, run *fn* in the thread executor, return 202 immediately."""
    job_id = start_job(name)

    def _run():
        try:
            result = fn(*args)
            finish_job(job_id, result if isinstance(result, dict) else {"count": result})
            logger.info("Job '%s' (id=%d) finished: %s", name, job_id, result)
        except Exception as exc:
            logger.error("Job '%s' (id=%d) failed: %s", name, job_id, exc)
            fail_job(job_id, str(exc))

    asyncio.get_event_loop().run_in_executor(_executor, _run)
    return {"status": "started", "job": name, "job_id": job_id}


# ── Job status endpoints ──────────────────────────────────────────────────────

@router.get("/job-status")
async def job_status():
    """Return the most recent graph job (running or finished)."""
    job = get_latest_job()
    if not job:
        return {"status": "no_jobs"}
    return job


@router.get("/job-history")
async def job_history(limit: int = 20):
    """Return the last *limit* graph jobs ordered newest-first."""
    return {"jobs": get_job_history(limit=limit)}


# ── Long-running jobs (202 Accepted) ─────────────────────────────────────────

@router.post("/build", status_code=202)
async def build_graph():
    """Full pipeline: build graph nodes → summaries → embeddings. Returns immediately."""
    _assert_idle()
    builder = get_graph_builder()
    return await _dispatch("build", builder.build_pipeline)


@router.post("/rebuild", status_code=202)
async def rebuild_graph():
    """Clear the graph then run the full pipeline: build → summaries → embeddings. Returns immediately."""
    _assert_idle()
    builder = get_graph_builder()

    def _rebuild():
        builder.clear_graph()
        return builder.build_pipeline()

    return await _dispatch("rebuild", _rebuild)


@router.post("/generate-embeddings", status_code=202)
async def generate_embeddings():
    """Generate embeddings for all nodes that lack them. Returns immediately."""
    _assert_idle()
    builder = get_graph_builder()
    return await _dispatch("generate-embeddings", builder.generate_embeddings)


@router.post("/generate-summaries", status_code=202)
async def generate_summaries():
    """Generate rich chapter/document summaries for top-down retrieval. Returns immediately."""
    _assert_idle()
    builder = get_graph_builder()
    return await _dispatch("generate-summaries", builder.generate_summaries)


@router.post("/compute-similarity", status_code=202)
async def compute_similarity(threshold: float = 0.80, top_k: int = 5):
    """Compute SEMANTICALLY_SIMILAR edges between sections. Returns immediately."""
    _assert_idle()
    builder = get_graph_builder()
    return await _dispatch(
        "compute-similarity",
        builder.compute_semantic_similarity,
        threshold,
        top_k,
    )


# ── Fast read-only endpoints ──────────────────────────────────────────────────

@router.get("/stats", response_model=GraphStatsResponse)
async def graph_stats() -> GraphStatsResponse:
    """Get current graph node and relationship counts."""
    try:
        querier = get_graph_querier()
        stats = querier.get_graph_stats()
        return GraphStatsResponse(stats=stats)
    except Exception as e:
        logger.error("Could not get graph stats: %s", e)
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
async def process_document_manually(filename: str, reparse_only: bool = False):
    """Manually trigger OCR pipeline for a document already in the documents folder.

    Set ``reparse_only=true`` to skip the Mistral OCR call and re-parse from
    the cached ``.md`` file.  Use this when only the parsing logic has changed
    (e.g. heading normalisation) and the raw OCR output is still valid.
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
        process_document_background(str(file_path), file_path.name, reparse_only=reparse_only)
        return {"status": "processing_started", "filename": file_path.name, "reparse_only": reparse_only}
    except Exception as e:
        logger.error("Failed to start processing: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-all-documents")
async def process_all_documents():
    """Trigger OCR pipeline for all PDFs in the documents folder."""
    from config.settings import settings
    docs_dir = Path(settings.documents_path)

    pdf_files = [p for p in docs_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
    if not pdf_files:
        return {"status": "no_documents", "message": "No PDF files found."}

    started: List[str] = []
    try:
        from backend.app.modules.ocr_pipeline import process_document_background
        for pdf_path in pdf_files:
            process_document_background(str(pdf_path), pdf_path.name)
            started.append(pdf_path.name)
    except Exception as e:
        logger.error("Failed to start batch processing: %s", e)

    return {"status": "processing_started", "documents": started, "count": len(started)}


class SemanticSearchResponse(BaseModel):
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
        logger.error("Semantic search failed: %s", e)
        return SemanticSearchResponse(query=query, results=[])
