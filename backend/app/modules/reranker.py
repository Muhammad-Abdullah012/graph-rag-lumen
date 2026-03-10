"""Cross-encoder re-ranker for Graph-RAG retrieval.

After RRF merges vector + BM25 results into ~8 candidate pages, the
cross-encoder scores each (query, page_content) pair and drops
low-relevance pages before they reach the LLM.

Long pages are split into overlapping chunks and scored independently.
The **maximum chunk score** is used as the page's relevance score so
that a relevant section anywhere in the page is not missed.

Model
=====
Default: ``BAAI/bge-reranker-v2-m3`` (multilingual, supports German).
Output: raw logits — positive = relevant, negative = irrelevant.
Threshold ``0.0`` is a sensible default.

Chunk size
==========
``CHUNK_SIZE = 4000`` chars with ``CHUNK_OVERLAP = 400`` chars.
bge-reranker-v2-m3 supports up to 8192 tokens (~32 000 chars for German),
so each chunk is well within the limit and keeps inference fast.
A typical OCR page (~3 000–8 000 chars) produces 1–2 chunks.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from config.settings import settings

logger = logging.getLogger(__name__)

_reranker_instance: Optional["Reranker"] = None

CHUNK_SIZE = 4000     # chars per chunk
CHUNK_OVERLAP = 400   # overlap between consecutive chunks


def _chunk_text(text: str) -> List[str]:
    """Split *text* into overlapping chunks of ``CHUNK_SIZE`` chars."""
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start: start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


class Reranker:
    """Thin wrapper around a ``sentence-transformers`` CrossEncoder."""

    def __init__(self) -> None:
        from sentence_transformers import CrossEncoder
        # Always run on CPU — GPU is reserved for Ollama (LLM + embeddings).
        # Cross-encoder scoring of ~10 pages is fast enough on CPU.
        self.model = CrossEncoder(
            settings.reranker_model,
            device="cpu",
        )
        logger.info("Reranker loaded: %s (device=cpu)", settings.reranker_model)

    def rerank(
        self,
        query: str,
        pages: List[Dict[str, Any]],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Score and reorder *pages* by relevance to *query*.

        Each page's content is split into overlapping chunks. All chunks
        are scored in one batched inference call. The max chunk score
        becomes the page score, ensuring relevant content anywhere in the
        page is captured.

        Returns at most *top_k* pages whose score is ≥ *threshold*,
        sorted by descending score.
        """
        if not pages:
            return pages

        # Build (query, chunk) pairs and track which page each belongs to.
        pairs: List[Tuple[str, str]] = []
        page_index: List[int] = []   # pairs[i] belongs to pages[page_index[i]]

        for idx, page in enumerate(pages):
            content = page.get("content") or ""
            for chunk in _chunk_text(content):
                pairs.append((query, chunk))
                page_index.append(idx)

        all_scores = self.model.predict(pairs)

        # Aggregate: max score per page across all its chunks.
        page_scores: List[float] = [float("-inf")] * len(pages)
        for score, pidx in zip(all_scores, page_index):
            if score > page_scores[pidx]:
                page_scores[pidx] = float(score)

        scored = sorted(
            zip(pages, page_scores),
            key=lambda x: x[1],
            reverse=True,
        )

        result: List[Dict[str, Any]] = []
        for page, score in scored[:top_k]:
            if score >= threshold:
                page["rerank_score"] = score
                result.append(page)

        logger.info(
            "Reranked %d pages (%d chunks) → %d kept (top=%.2f, threshold=%.2f)",
            len(pages), len(pairs), len(result),
            max(page_scores) if page_scores else 0.0,
            threshold,
        )
        return result


def get_reranker() -> Optional[Reranker]:
    """Return the singleton Reranker, or ``None`` if disabled / unavailable."""
    global _reranker_instance
    if not settings.reranker_enabled:
        return None
    if _reranker_instance is None:
        try:
            _reranker_instance = Reranker()
        except Exception as e:
            logger.error("Failed to load reranker: %s", e)
            return None
    return _reranker_instance
