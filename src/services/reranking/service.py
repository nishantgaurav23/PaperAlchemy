"""Cross-encoder re-ranking service (S4b.1).

Re-scores search results using a cross-encoder model for more accurate
relevance ranking. Supports local (sentence-transformers) and cloud (Cohere) providers.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import numpy as np
from src.config import RerankerSettings
from src.exceptions import RerankerError
from src.schemas.api.search import SearchHit

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """A single re-ranked document with its relevance score."""

    index: int
    relevance_score: float
    document: dict


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply sigmoid to normalize raw logits to 0.0-1.0."""
    return 1.0 / (1.0 + np.exp(-x))


def _extract_text(hit: SearchHit) -> str:
    """Extract text from a SearchHit: chunk_text > abstract > title."""
    return hit.chunk_text or hit.abstract or hit.title


class RerankerService:
    """Re-ranks documents using a cross-encoder model or Cohere API."""

    def __init__(
        self,
        settings: RerankerSettings,
        model: object | None = None,
        provider: str = "local",
    ) -> None:
        self._settings = settings
        self._model = model
        self._provider = provider

    async def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Re-rank documents by relevance to the query.

        Args:
            query: The search query.
            documents: List of dicts with at least a 'text' key.
            top_k: Number of top results to return (default: settings.top_k).

        Returns:
            List of RerankResult sorted by relevance_score descending.
        """
        if not documents:
            return []

        if top_k is None:
            top_k = self._settings.top_k

        if self._provider == "cohere":
            return await self._rerank_cohere(query, documents, top_k)

        return await self._rerank_local(query, documents, top_k)

    async def _rerank_local(
        self,
        query: str,
        documents: list[dict],
        top_k: int,
    ) -> list[RerankResult]:
        """Re-rank using local cross-encoder model."""
        # Separate documents with and without text
        scorable_indices = []
        empty_indices = []
        pairs = []

        for i, doc in enumerate(documents):
            text = doc.get("text", "")
            if text:
                scorable_indices.append(i)
                pairs.append([query, text])
            else:
                empty_indices.append(i)

        results: list[RerankResult] = []

        if pairs:
            # Run predict in thread pool to avoid blocking the async loop
            loop = asyncio.get_event_loop()
            raw_scores = await loop.run_in_executor(None, self._model.predict, pairs)
            normalized = _sigmoid(np.array(raw_scores, dtype=float))

            for idx, score in zip(scorable_indices, normalized, strict=True):
                results.append(RerankResult(index=idx, relevance_score=float(score), document=documents[idx]))

        # Empty-text documents get score 0.0
        for idx in empty_indices:
            results.append(RerankResult(index=idx, relevance_score=0.0, document=documents[idx]))

        # Sort by score descending, truncate to top_k
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:top_k]

    async def _rerank_cohere(
        self,
        query: str,
        documents: list[dict],
        top_k: int,
    ) -> list[RerankResult]:
        """Re-rank using Cohere Rerank API."""
        import httpx

        doc_texts = [doc.get("text", "") for doc in documents]

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.cohere.com/v2/rerank",
                    headers={
                        "Authorization": f"Bearer {self._settings.cohere_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._settings.model,
                        "query": query,
                        "documents": doc_texts,
                        "top_n": top_k,
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()
        except Exception as e:
            raise RerankerError(f"Cohere rerank failed: {e}") from e

        results = []
        for item in data.get("results", []):
            idx = item["index"]
            results.append(
                RerankResult(
                    index=idx,
                    relevance_score=float(item["relevance_score"]),
                    document=documents[idx],
                )
            )

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:top_k]

    async def rerank_search_hits(
        self,
        query: str,
        hits: list[SearchHit],
        top_k: int | None = None,
    ) -> list[SearchHit]:
        """Re-rank SearchHit objects and return them with updated scores.

        Text extraction priority: chunk_text > abstract > title.
        Hits with no text get score 0.0.
        """
        if not hits:
            return []

        if top_k is None:
            top_k = self._settings.top_k

        # Build document dicts for reranking
        documents = []
        for hit in hits:
            text = _extract_text(hit)
            documents.append({"text": text, "hit_index": len(documents)})

        reranked = await self.rerank(query, documents, top_k=top_k)

        # Map back to SearchHit objects with updated scores
        result_hits = []
        for r in reranked:
            original_hit = hits[r.index]
            updated = original_hit.model_copy(update={"score": r.relevance_score})
            result_hits.append(updated)

        return result_hits
