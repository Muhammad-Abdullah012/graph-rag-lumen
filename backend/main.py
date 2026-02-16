"""Main FastAPI Application"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api import qa, health, graph, conversations, files
from backend.app.modules.database import (
    get_neo4j_connection,
    init_postgres_pool,
    close_postgres_pool,
)
from backend.app.modules.graph_builder import get_graph_builder
from backend.app.modules.agent import init_agent
from backend.app.modules.embeddings import get_embedding_service
from config.settings import settings
from config.logging_config import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    try:
        # 1. Neo4j
        logger.info("Initializing Neo4j connection ...")
        get_neo4j_connection()

        # 2. PostgreSQL (async pool + schema)
        logger.info("Initializing PostgreSQL pool ...")
        await init_postgres_pool()

        # 3. Build / refresh knowledge graph from JSON files
        logger.info("Building knowledge graph from JSON files ...")
        builder = get_graph_builder()
        stats = builder.build_all()
        logger.info(f"Graph build result: {stats}")

        # 4. Sync embeddings (Neo4j → pgvector)
        logger.info("Syncing graph embeddings to pgvector ...")
        embed_svc = get_embedding_service()
        embed_stats = await embed_svc.sync_graph_embeddings()
        logger.info(f"Embedding sync result: {embed_stats}")

        # 5. Initialise Agent (checkpointer + tools)
        logger.info("Initializing LangGraph agent ...")
        await init_agent()

        # 6. Ensure upload directory exists
        Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)

        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}", exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("Shutting down ...")
    await close_postgres_pool()
    logger.info("Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Eurocode Graph RAG API",
    description="Knowledge Graph Q&A System for Eurocode standards with semantic search and persistent conversations",
    version="3.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router)
app.include_router(qa.router)
app.include_router(graph.router)
app.include_router(conversations.router)
app.include_router(files.router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Eurocode Graph RAG API",
        "version": "3.0.0",
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
