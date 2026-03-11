"""Embedding service module — Jina AI vector generation."""

from src.services.embeddings.client import JinaEmbeddingsClient
from src.services.embeddings.factory import make_embeddings_client

__all__ = ["JinaEmbeddingsClient", "make_embeddings_client"]
