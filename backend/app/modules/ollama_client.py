"""Ollama LLM Integration"""
import logging
import time
from typing import Optional
import requests

from config.settings import settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama API"""
    def __init__(self):
        """Initialize Ollama client"""
        self.base_url = settings.ollama_base_url.rstrip('/')
        self._verify_connectivity()
    def _verify_connectivity(self):
        """Verify Ollama service is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            logger.info("Connected to Ollama successfully")
        except Exception as e:
            logger.error(f"Could not connect to Ollama at {self.base_url}: {str(e)}")
            raise
    def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
    ) -> str:
        """
        Generate text using LLM
        Args:
            prompt: Input prompt
            model: Model to use (defaults to settings.ollama_llm_model)
            temperature: Temperature for generation
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
        Returns:
            Generated text
        """
        if model is None:
            model = settings.ollama_llm_model
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "stream": False,
                },
                timeout=300,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise

    def generate_embedding(self, text: str, model: Optional[str] = None) -> list:
        """Generate an embedding vector for the given text.

        Retries up to 3 times with exponential backoff (2 s, 4 s, 8 s) on
        timeout errors.  Other transient errors are also retried but logged
        at ERROR level.  Returns an empty list only after all attempts fail.

        Args:
            text: Input text to embed.
            model: Embedding model (defaults to settings.ollama_embedding_model).

        Returns:
            List of floats representing the embedding vector, or [] on failure.
        """
        if model is None:
            model = getattr(settings, "ollama_embedding_model", "nomic-embed-text")

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/api/embed",
                    json={"model": model, "input": text},
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()
                embeddings = data.get("embeddings", [])
                if embeddings:
                    return embeddings[0]
                # Fallback: older Ollama API format
                return data.get("embedding", [])
            except requests.exceptions.Timeout:
                wait = 2 ** attempt  # 2s, 4s, 8s
                logger.warning(
                    "Embedding request timed out (attempt %d/%d). Retrying in %ds…",
                    attempt, max_retries, wait,
                )
                if attempt < max_retries:
                    time.sleep(wait)
            except Exception as e:
                logger.error(
                    "Error generating embedding (attempt %d/%d): %s",
                    attempt, max_retries, e,
                )
                if attempt < max_retries:
                    time.sleep(2 ** attempt)

        logger.error(
            "Embedding failed after %d attempts for text: %r", max_retries, text[:80]
        )
        return []

    def list_models(self) -> list:
        """List available models in Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []


# Singleton instance
_ollama_client = None


def get_ollama_client() -> OllamaClient:
    """Get or create Ollama client"""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client
