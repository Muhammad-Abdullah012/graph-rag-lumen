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
    
    # Indexes — Page
    vector_index_name: str = os.getenv("VECTOR_INDEX_NAME", "page_embeddings")
    fulltext_index_name: str = os.getenv("FULLTEXT_INDEX_NAME", "page_fulltext")
    # Indexes — Reference
    reference_vector_index_name: str = os.getenv("REFERENCE_VECTOR_INDEX_NAME", "reference_embeddings")
    reference_fulltext_index_name: str = os.getenv("REFERENCE_FULLTEXT_INDEX_NAME", "reference_fulltext")
    # Indexes — Table
    table_vector_index_name: str = os.getenv("TABLE_VECTOR_INDEX_NAME", "table_embeddings")
    table_fulltext_index_name: str = os.getenv("TABLE_FULLTEXT_INDEX_NAME", "table_fulltext")
    # Indexes — Image
    image_vector_index_name: str = os.getenv("IMAGE_VECTOR_INDEX_NAME", "image_embeddings")
    image_fulltext_index_name: str = os.getenv("IMAGE_FULLTEXT_INDEX_NAME", "image_fulltext")

    # LLM context window
    ollama_num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
    # Per-chunk read timeout for streaming Ollama calls (seconds)
    ollama_chat_timeout: int = int(os.getenv("OLLAMA_CHAT_TIMEOUT", "300"))

    # Retrieval
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    retrieval_neighbor_pages: int = int(os.getenv("RETRIEVAL_NEIGHBOR_PAGES", "2"))
    retrieval_neighbor_expand_top: int = int(os.getenv("RETRIEVAL_NEIGHBOR_EXPAND_TOP", "3"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))
    
    # Mistral OCR
    mistral_api_key: str = os.getenv("MISTRAL_API_KEY", "")
    json_output_path: str = os.getenv("JSON_OUTPUT_PATH", "/backend/json")
    images_path: str = os.getenv("IMAGES_PATH", "/backend/images")
    batch_poll_interval: int = int(os.getenv("BATCH_POLL_INTERVAL", "5"))

    # Agentic retrieval (2-call: draft → extract refs → fetch → final)
    reference_extraction_max_refs: int = int(os.getenv("REFERENCE_EXTRACTION_MAX_REFS", "5"))
    reference_extraction_max_pages: int = int(os.getenv("REFERENCE_EXTRACTION_MAX_PAGES", "3"))
    reference_neighbor_pages: int = int(os.getenv("REFERENCE_NEIGHBOR_PAGES", "1"))

    # Reference / Table / Image node extraction
    reference_context_chars: int = int(os.getenv("REFERENCE_CONTEXT_CHARS", "100"))
    reference_resolve_score_threshold: float = float(os.getenv("REFERENCE_RESOLVE_SCORE_THRESHOLD", "2.0"))

    # Context budget
    chars_per_token: int = int(os.getenv("CHARS_PER_TOKEN", "4"))
    context_reserved_tokens: int = int(os.getenv("CONTEXT_RESERVED_TOKENS", "500"))

    # SSE keepalive: emit a status ping every N tokens during draft generation
    # to prevent proxies/browsers from closing the idle connection
    draft_keepalive_interval: int = int(os.getenv("DRAFT_KEEPALIVE_INTERVAL", "30"))

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
