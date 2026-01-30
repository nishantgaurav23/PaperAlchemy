"""
Jina AI Embeddings Client — async, batched vector generation.

Why it's needed:
    Hybrid search requires vector representations of text. The Jina client
    converts text passages and search queries into 1024-dimensional vectors
    that capture semantic meaning. Without embeddings, we can only do keyword
    search (BM25), which misses conceptually similar results that use
    different terminology.

What it does:
    - embed_passages(): Converts document chunks into vectors for indexing.
      Processes in batches (default 100) to stay within API limits and
      maximize throughput. Uses task="retrieval.passage" for document-side
      embeddings.
    - embed_query(): Converts a single search query into a vector.
      Uses task="retrieval.query" which produces embeddings optimized for
      matching against passage embeddings (asymmetric encoding).
    - Async HTTP via httpx for non-blocking I/O in FastAPI's event loop.

How it helps:
    - Semantic search: "machine learning" matches "deep neural networks"
    - Batched processing: 100 chunks per API call → fewer round trips
    - Async: doesn't block other requests while waiting for Jina API
    - Context manager: ensures HTTP client is properly closed on shutdown

Architecture:
    JinaEmbeddingsClient is created by the factory (factory.py) using
    settings.jina_api_key. It's injected into HybridIndexer for indexing
    and into the hybrid_search router for query-time embedding.
"""

import logging
from typing import List

import httpx

from src.schemas.embeddings.jina import JinaEmbeddingRequest, JinaEmbeddingResponse

logger = logging.getLogger(__name__)


class JinaEmbeddingsClient:
    """Async client for Jina AI embeddings API.

    Uses Jina embeddings v3 model with 1024 dimensions optimized for
    retrieval tasks. Supports both passage (document) and query embeddings
    with asymmetric encoding for better search accuracy.

    Usage:
        # As async context manager (recommended)
        async with JinaEmbeddingsClient(api_key="...") as client:
            vectors = await client.embed_passages(["text1", "text2"])
            query_vec = await client.embed_query("search term")

        # Or manually
        client = JinaEmbeddingsClient(api_key="...")
        vectors = await client.embed_passages(["text1", "text2"])
        await client.close()
    """

    def __init__(self, api_key: str, base_url: str = "https://api.jina.ai/v1"):
        """Initialize Jina embeddings client.

        Args:
            api_key: Jina API key from https://jina.ai/embeddings.
                     Free tier allows 1M tokens/month — sufficient for
                     thousands of paper chunks.
            base_url: Jina API base URL. Override for testing or
                      self-hosted Jina instances.
        """
        # Store API key for authentication header
        self.api_key = api_key

        # Base URL for all API calls (e.g., POST {base_url}/embeddings)
        self.base_url = base_url

        # HTTP headers sent with every request:
        # - Authorization: Bearer token for API authentication
        # - Content-Type: tells Jina we're sending JSON
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # httpx.AsyncClient is the async HTTP client that handles:
        # - Connection pooling (reuses TCP connections across requests)
        # - Automatic retries on connection errors
        # - 30-second timeout to avoid hanging on slow responses
        self.client = httpx.AsyncClient(timeout=30.0)

        logger.info("Jina embeddings client initialized")

    async def embed_passages(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Embed text passages for indexing into OpenSearch.

        Processes texts in batches to maximize throughput while staying
        within Jina's API limits (max 2048 texts per request).

        Args:
            texts: List of text passages to embed. Each passage is
                   typically a chunk from TextChunker (400-800 words).
            batch_size: Number of texts per API call. Default 100 balances
                        throughput vs memory. Jina max is 2048.

        Returns:
            List of embedding vectors (each is List[float] of length 1024).
            Order matches input texts — embeddings[i] corresponds to texts[i].

        Raises:
            httpx.HTTPError: On API errors (auth failure, rate limit, etc.)

        Example:
            texts = ["chunk 1 text...", "chunk 2 text..."]
            vectors = await client.embed_passages(texts)
            # vectors[0] is the 1024-dim vector for "chunk 1 text..."
        """
        # Accumulates all embedding vectors across batches
        embeddings = []

        # Process texts in batches of batch_size
        # range(0, len(texts), batch_size) produces: [0, 100, 200, ...]
        for i in range(0, len(texts), batch_size):
            # Slice the current batch from the full list
            batch = texts[i: i + batch_size]

            # Build the Jina API request using our Pydantic model
            # task="retrieval.passage" optimizes embeddings for documents
            # (as opposed to "retrieval.query" which optimizes for queries)
            request_data = JinaEmbeddingRequest(
                model="jina-embeddings-v3",
                task="retrieval.passage",
                dimensions=1024,
                input=batch
            )

            try:
                # POST to Jina's /embeddings endpoint
                # .model_dump() converts Pydantic model to dict for JSON serialization
                response = await self.client.post(
                    f"{self.base_url}/embeddings",
                    headers=self.headers,
                    json=request_data.model_dump()
                )
                # Raises httpx.HTTPStatusError on 4xx/5xx responses
                response.raise_for_status()

                # Parse response using our Pydantic model for validation
                result = JinaEmbeddingResponse(**response.json())

                # Extract just the embedding vectors from the response data
                # Each item in result.data has: {"object": "embedding", "index": 0, "embedding": [...]}
                batch_embeddings = [item["embedding"] for item in result.data]
                embeddings.extend(batch_embeddings)

                logger.debug(f"Embedded batch of {len(batch)} passages")

            except httpx.HTTPError as e:
                logger.error(f"Error embedding passages: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error in embed_passages: {e}")
                raise

        logger.info(f"Successfully embedded {len(texts)} passages")
        return embeddings

    async def embed_query(self, query: str) -> List[float]:
        """Embed a search query for vector similarity search.

        Uses task="retrieval.query" which produces embeddings optimized
        for matching against passage embeddings. This asymmetric encoding
        improves retrieval accuracy compared to using the same task type
        for both queries and documents.

        Args:
            query: Search query text (e.g., "transformer attention mechanism").

        Returns:
            Single embedding vector (List[float] of length 1024).

        Raises:
            httpx.HTTPError: On API errors.

        Example:
            query_vec = await client.embed_query("attention mechanism")
            # Use query_vec for KNN search in OpenSearch
        """
        # Build request with task="retrieval.query" for query-side embedding
        # input is a list even for single queries (API requirement)
        request_data = JinaEmbeddingRequest(
            model="jina-embeddings-v3",
            task="retrieval.query",
            dimensions=1024,
            input=[query]
        )

        try:
            response = await self.client.post(
                f"{self.base_url}/embeddings",
                headers=self.headers,
                json=request_data.model_dump()
            )
            response.raise_for_status()

            result = JinaEmbeddingResponse(**response.json())
            # Extract the single embedding from the response
            embedding = result.data[0]["embedding"]

            logger.debug(f"Embedded query: '{query[:50]}...'")
            return embedding

        except httpx.HTTPError as e:
            logger.error(f"Error embedding query: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in embed_query: {e}")
            raise

    async def close(self):
        """Close the HTTP client and release connection pool resources.

        Should be called when the client is no longer needed.
        Automatically called when using async context manager.
        """
        await self.client.aclose()

    async def __aenter__(self):
        """Async context manager entry — returns self for use in 'async with'."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit — ensures HTTP client is closed."""
        await self.close()
