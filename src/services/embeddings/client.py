"""Async client for Jina AI Embeddings API.

Converts text into 1024-dimensional vectors for hybrid search.
Supports asymmetric encoding: retrieval.passage for documents,
retrieval.query for search queries.
"""

from __future__ import annotations

import logging

import httpx
from src.exceptions import EmbeddingServiceError
from src.schemas.embeddings import JinaEmbeddingRequest, JinaEmbeddingResponse

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.jina.ai/v1"


class JinaEmbeddingsClient:
    """Async Jina AI embeddings client with batch support."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "jina-embeddings-v3",
        dimensions: int = 1024,
        base_url: str = _BASE_URL,
        timeout: int = 30,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions
        self._base_url = base_url
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.AsyncClient(timeout=float(timeout))
        self._closed = False

    async def embed_passages(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """Embed text passages for indexing. Returns list of 1024-dim vectors."""
        if not texts:
            return []

        embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = await self._call_api(batch, task="retrieval.passage")
            embeddings.extend([item.embedding for item in result.data])

        logger.info("Embedded %d passages", len(texts))
        return embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single search query. Returns 1024-dim vector."""
        if not query or not query.strip():
            raise ValueError("Query must not be empty")

        result = await self._call_api([query], task="retrieval.query")
        return result.data[0].embedding

    async def _call_api(self, texts: list[str], *, task: str) -> JinaEmbeddingResponse:
        """Make a single API call to Jina embeddings endpoint."""
        request = JinaEmbeddingRequest(
            model=self._model,
            task=task,
            dimensions=self._dimensions,
            input=texts,
        )

        try:
            response = await self._client.post(
                f"{self._base_url}/embeddings",
                headers=self._headers,
                json=request.model_dump(),
            )
            response.raise_for_status()
            return JinaEmbeddingResponse(**response.json())

        except httpx.TimeoutException as exc:
            raise EmbeddingServiceError(
                detail=f"Timeout calling Jina API: {exc}",
            ) from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 401:
                msg = "Authentication failed — check JINA__API_KEY"
            elif status == 429:
                msg = "Rate limit exceeded — reduce request frequency"
            else:
                msg = f"Jina API returned HTTP {status}"
            raise EmbeddingServiceError(detail=msg) from exc
        except httpx.HTTPError as exc:
            raise EmbeddingServiceError(
                detail=f"Jina API request failed: {exc}",
            ) from exc

    async def close(self) -> None:
        """Close the HTTP client. Safe to call multiple times."""
        if not self._closed:
            await self._client.aclose()
            self._closed = True

    async def __aenter__(self) -> JinaEmbeddingsClient:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
