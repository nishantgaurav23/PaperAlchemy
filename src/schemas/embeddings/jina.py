"""
Why it's needed:
    Before we can clal the JINA API, we need Pydantic models that validate the request we send and the response we receive.
    This Prevents sending malformed requests and ensures we parse response correctly. Pydantic also gives us automatic serialization
    via .model_dump()

What it does:
    - JinaEmbeddingRequest - strutures the outgoing API call with model name, task type (passage vs query), dimensions, and input
    texts
    - JinaEmbeddingResponse - parses the API response, extracting usage stats and embedding vectors

How it helps:
    Type safety at the API response boundary. If JINA changes their response format, validation fails immediately with a clear
    error instead of silently producing wrong embeddings.

Pydantic models for Jina AI Embeddings API

These models define the exact shape of requests sent to and responses
received from the Jina embeddings API (https://api.jina.ai/v1/embeddings).

Jina v3 supports two task types that produce different embeddings:
    - "retrieval.passage": Optimized for documents being indexed.
    - "retrieval.query": Optimized for search queries

Using the correct task type improves retrieval acuracy because Jina
trains separate projections for queries vs passages.

References: https://jina.ai.embeddings
"""

from typing import Dict, List

from pydantic import BaseModel

class JinaEmbeddingRequest(BaseModel):
    """
    Request payload sent to jina's /v1/embeddings endpoint

    Attributes:
        model: Jina model identifier. We use "jina-embedding-v3" which 
               supports up to 8192 tokens and multiple task types.
        task: Embedding task type. Use "retrieval.passage" when indexing
              documents and "retrieval.query" when embedding search queries.
              This asymetric encoding imporoves search quality.
        dimensions: Output vector dimensions. 1024 gives the best accuracy
                for JINA v3. Lower values (256, 512) trade accuracy for speed
                and storage.
        late_chunking: If True, Jina handles chunking internally.We set
                False because we do our own section-aware chunking
                in TextChunker for better control.
        embedding_type: Vector element type,. "float" gives full 32-bit
                precision. "binary" or "ubinary" are smaller but less accurate.
        input: List of text strings to embed. Maximum 2048 texts per
               requests. Each text can be up to 8192 tokens.
    """

    model: str = "jina-embeddings-v3"
    task: str = "retrieval.passage"
    dimensions: int = 1024
    late_chunking: bool = False
    embedding_type: str = "float"
    input: List[str]

class JinaEmbeddingResponse(BaseModel):
    """Response received from Jina's /v1/embeddings endpoint.

    Attributes:
        model: Echo of the model used for embedding.
        object: Response type, always "list" for embedding responses.
        usage: Token usage statistics with keys:
                - "total tokens": Total tokens processed.
                - "prompt_tokens": Tokens in the input texts
                Useful for tracking API costs ($0.02 per 1M tokens).
        data: List of embedding results. Each entry is a dict with:
                - "object": "embedding"
                - "index": Position i nthe input list (0-based)
                - "embedding": List[float] of length `dimensions`
    """

    model: str
    object: str = "list"
    usage: Dict[str, int]
    data: List[Dict]

