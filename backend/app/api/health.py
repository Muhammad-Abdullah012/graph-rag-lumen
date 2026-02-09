"""Health Check Routes"""
import logging
from fastapi import APIRouter
from pydantic import BaseModel

from backend.app.modules.database import get_neo4j_connection
from backend.app.modules.ollama_client import get_ollama_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/health", tags=["health"])


class HealthStatus(BaseModel):
    """Health check response"""
    status: str
    neo4j: str
    ollama: str


@router.get("/", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """
    Check application health
    
    Returns:
        Health status
    """
    neo4j_status = "ok"
    ollama_status = "ok"
    
    try:
        db = get_neo4j_connection()
        db.execute_query("RETURN 1")
    except Exception as e:
        logger.error(f"Neo4j health check failed: {str(e)}")
        neo4j_status = "error"
    
    try:
        ollama = get_ollama_client()
        ollama.list_models()
    except Exception as e:
        logger.error(f"Ollama health check failed: {str(e)}")
        ollama_status = "error"
    
    overall_status = "ok" if neo4j_status == "ok" and ollama_status == "ok" else "degraded"
    
    return HealthStatus(
        status=overall_status,
        neo4j=neo4j_status,
        ollama=ollama_status
    )
