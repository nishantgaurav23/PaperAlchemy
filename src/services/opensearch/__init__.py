"""OpenSearch service module."""

from .client import OpenSearchClient
from .factory import make_opensearch_client, make_opensearch_client_fresh
from .query_builder import QueryBuilder
from .index_config import (
    ARXIV_PAPERS_INDEX,
    ARXIV_PAPERS_MAPPING,
    ARXIV_PAPERS_CHUNKS_INDEX,
    ARXIV_PAPERS_CHUNKS_MAPPING,
    HYBRID_RRF_PIPELINE
)

__all__ = [
    "OpenSearchClient",
    "make_opensearch_client",
    "make_opensearch_client_fresh",
    "QueryBuilder",
    "ARXIV_PAPERS_INDEX",
    "ARXIV_PAPERS_MAPPING",
    "ARXIV_PAPERS_CHUNKS_INDEX",
    "ARXIV_PAPERS_CHUNKS_MAPPING",
    "HYBRID_RRF_PIPELINE"
]