"""Main FastAPI Application"""
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api import documents, graph, health, qa
from backend.app.modules.database import get_neo4j_connection
from backend.app.modules.graph_builder import get_graph_builder
from backend.app.modules.graph_jobs_db import fail_job, finish_job, start_job
from config.logging_config import logger
from config.settings import settings

# Directory for OCR-extracted images
IMAGES_DIR = str(Path(__file__).parent / "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

_startup_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="startup_worker")


def _graph_node_count() -> int:
    """Return the current number of nodes in Neo4j, or 0 on error."""
    try:
        conn = get_neo4j_connection()
        rows = conn.execute_query("MATCH (n) RETURN count(n) AS c")
        return rows[0]["c"] if rows else 0
    except Exception:
        return 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    logger.info("Initializing database connection...")
    try:
        get_neo4j_connection()
    except Exception as e:
        logger.error("Failed to connect to Neo4j: %s", e)
        raise

    node_count = _graph_node_count()

    if node_count > 0:
        # Graph already populated — skip rebuild, start immediately.
        logger.info("Graph already contains %d nodes — skipping auto-build.", node_count)
    else:
        # First run or empty graph — trigger the full pipeline in the background.
        logger.info("Graph is empty — triggering initial build pipeline in background...")

        def _run_initial_build():
            job_id = start_job("initial-build")
            try:
                builder = get_graph_builder()
                result = builder.build_pipeline()
                finish_job(job_id, result)
                logger.info("Initial build pipeline complete: %s", result)
            except Exception as exc:
                logger.error("Initial build pipeline failed: %s", exc)
                fail_job(job_id, str(exc))

        loop = asyncio.get_event_loop()
        loop.run_in_executor(_startup_executor, _run_initial_build)

    logger.info("Application started successfully")
    yield
    logger.info("Application shutdown")


# Create FastAPI app
app = FastAPI(
    title="Eurocode Graph RAG API",
    description="Knowledge Graph Q&A System for Eurocode standards",
    version="2.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(qa.router)
app.include_router(graph.router)
app.include_router(documents.router)

# Serve stored documents as static files
app.mount(
    "/documents",
    StaticFiles(directory=documents.DOCUMENTS_DIR),
    name="documents",
)

# Serve OCR-extracted images as static files
app.mount(
    "/api/images",
    StaticFiles(directory=IMAGES_DIR),
    name="images",
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Eurocode Graph RAG API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.backend_host,
        port=settings.backend_port,
        log_level=settings.log_level.lower(),
    )
