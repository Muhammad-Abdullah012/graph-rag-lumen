"""Application Settings and Configuration"""
import os
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
    ollama_llm_model: str = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
    ollama_embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

    # Mistral OCR
    mistral_api_key: str = os.getenv("MISTRAL_API_KEY", "")

    # PostgreSQL (processing status)
    postgres_host: str = os.getenv("POSTGRES_HOST", "postgres-graphrag")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_user: str = os.getenv("POSTGRES_USER", "graphrag")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "graphrag_secret")
    postgres_db: str = os.getenv("POSTGRES_DB", "graphrag")

    # Backend
    backend_host: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    backend_port: int = int(os.getenv("BACKEND_PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    environment: str = os.getenv("ENVIRONMENT", "development")

    # Documents
    documents_path: str = os.getenv("DOCUMENTS_PATH", "documents")
    max_upload_size: int = int(os.getenv("MAX_UPLOAD_SIZE", str(50 * 1024 * 1024)))

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


settings = Settings()
