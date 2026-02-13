"""Graph Management Routes - Build & inspect the knowledge graph"""
import logging
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any

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
