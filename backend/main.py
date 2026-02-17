"""Main FastAPI Application"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api import documents, graph, health, qa
from backend.app.modules.database import get_neo4j_connection
from backend.app.modules.graph_builder import get_graph_builder
from config.settings import settings
from config.logging_config import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    try:
        logger.info("Initializing database connection...")
        get_neo4j_connection()

        # Auto-build graph from JSON files on startup
        logger.info("Building knowledge graph from JSON files...")
        builder = get_graph_builder()
        stats = builder.build_all()
        logger.info(f"Graph build result: {stats}")

        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise

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
