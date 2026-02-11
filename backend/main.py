"""Main FastAPI Application"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch

from backend.app.api import documents, qa, health
from backend.app.modules.database import get_neo4j_connection
from config.settings import settings
from config.logging_config import logger

# Initialize database on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    try:
        logger.info("Initializing database connection...")
        get_neo4j_connection()
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision('high')
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise
    
    yield
    
    logger.info("Application shutdown")


# Create FastAPI app
app = FastAPI(
    title="Graph RAG API",
    description="Knowledge Graph RAG System with local LLMs",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(qa.router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Graph RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host=settings.backend_host,
        port=settings.backend_port,
        log_level=settings.log_level.lower()
    )
