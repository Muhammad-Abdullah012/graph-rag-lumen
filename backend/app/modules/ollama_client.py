"""Ollama LLM Integration"""
import json
import logging
from typing import Optional, Dict, Any, Generator
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
    
    def generate_embedding(self, text: str) -> list:
        """
        Generate embedding for text using embedding model
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={
                    "model": settings.ollama_embedding_model,
                    "input": text
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            return data.get("embeddings", [[]])[0]
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
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
                    "stream": False
                },
                timeout=300
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise
    
    def chat_stream(
        self,
        messages: list,
        model: Optional[str] = None,
        temperature: float = 0.1,
    ) -> Generator[str, None, None]:
        """Stream chat response token by token using /api/chat."""
        if model is None:
            model = settings.ollama_llm_model

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": temperature,
                        "num_ctx": settings.ollama_num_ctx,
                    },
                },
                stream=True,
                timeout=300,
            )
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break
        except Exception as e:
            logger.error(f"Error in chat stream: {str(e)}")
            raise

    def list_models(self) -> list:
        """List available models in Ollama"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10
            )
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
