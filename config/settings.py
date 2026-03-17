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
    
    # Indexes
    vector_index_name: str = os.getenv("VECTOR_INDEX_NAME", "page_embeddings")
    fulltext_index_name: str = os.getenv("FULLTEXT_INDEX_NAME", "page_fulltext")

    # LLM context window
    ollama_num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", "16384"))

    # Retrieval
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    
    # Mistral OCR
    mistral_api_key: str = os.getenv("MISTRAL_API_KEY", "")
    json_output_path: str = os.getenv("JSON_OUTPUT_PATH", "/backend/json")
    images_path: str = os.getenv("IMAGES_PATH", "/backend/images")
    batch_poll_interval: int = int(os.getenv("BATCH_POLL_INTERVAL", "5"))

    # Debug
    debug_log_path: str = os.getenv("DEBUG_LOG_PATH", "/app/debug_raw.jsonl")

    # Application
    max_upload_size: int = int(os.getenv("MAX_UPLOAD_SIZE", "52428800"))  # 50MB
    upload_chunk_size: int = int(os.getenv("UPLOAD_CHUNK_SIZE", "1048576"))  # 1MB
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


settings = Settings()
