"""Application Settings and Configuration"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application Settings loaded from environment variables"""
    
    # Neo4j
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    neo4j_username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")
    
    # Ollama
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest")
    ollama_llm_model: str = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
    ollama_graph_model: str = os.getenv("OLLAMA_GRAPH_MODEL", "llama3.4")
    
    # Backend
    backend_host: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    backend_port: int = int(os.getenv("BACKEND_PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    environment: str = os.getenv("ENVIRONMENT", "development")
    
    # Documents
    documents_path: str = os.getenv("DOCUMENTS_PATH", "/tmp/documents")
    documents_base_url: str = os.getenv("DOCUMENTS_BASE_URL", "http://localhost:8080/documents")
    
    # Vector Index
    vector_index_name: str = os.getenv("VECTOR_INDEX_NAME", "document_embeddings")
    vector_dimension: int = int(os.getenv("VECTOR_DIMENSION", "768"))
    
    # Application
    max_upload_size: int = int(os.getenv("MAX_UPLOAD_SIZE", "52428800"))  # 50MB
    pdf_extract_timeout: int = int(os.getenv("PDF_EXTRACT_TIMEOUT", "300"))
    graph_build_timeout: int = int(os.getenv("GRAPH_BUILD_TIMEOUT", "600"))
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
