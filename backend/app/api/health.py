"""Health Check Routes"""
import logging
from fastapi import APIRouter
from pydantic import BaseModel

from backend.app.modules.database import get_neo4j_connection, get_pg_pool
from backend.app.modules.ollama_client import get_ollama_client
from backend.app.modules.graph_querier import get_graph_querier

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/health", tags=["health"])


class HealthStatus(BaseModel):
    """Health check response"""
    status: str
    neo4j: str
    postgres: str
    ollama: str
    graph_loaded: bool


@router.get("/", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """Check application health"""
    neo4j_status = "ok"
    postgres_status = "ok"
    ollama_status = "ok"
    graph_loaded = False

    # Neo4j
    try:
        db = get_neo4j_connection()
        db.execute_query("RETURN 1")
    except Exception as e:
        logger.error(f"Neo4j health check failed: {str(e)}")
        neo4j_status = "error"

    # PostgreSQL
    try:
        pool = get_pg_pool()
        async with pool.connection() as conn:
            await conn.execute("SELECT 1")
    except Exception as e:
        logger.error(f"PostgreSQL health check failed: {str(e)}")
        postgres_status = "error"

    # Ollama
    try:
        ollama = get_ollama_client()
        ollama.list_models()
    except Exception as e:
        logger.error(f"Ollama health check failed: {str(e)}")
        ollama_status = "error"

    # Graph loaded?
    try:
        querier = get_graph_querier()
        stats = querier.get_graph_stats()
        graph_loaded = stats.get("symbols", 0) > 0
    except Exception:
        pass

    all_ok = neo4j_status == "ok" and postgres_status == "ok" and ollama_status == "ok"
    overall_status = "ok" if all_ok else "degraded"

    return HealthStatus(
        status=overall_status,
        neo4j=neo4j_status,
        postgres=postgres_status,
        ollama=ollama_status,
        graph_loaded=graph_loaded,
    )
