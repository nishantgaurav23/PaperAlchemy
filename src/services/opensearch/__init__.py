"""OpenSearch service: client, index config, query builder, and factories."""

from .client import OpenSearchClient
from .factory import make_opensearch_client, make_opensearch_client_fresh
from .index_config import ARXIV_PAPERS_CHUNKS_MAPPING, HYBRID_RRF_PIPELINE
from .query_builder import QueryBuilder

__all__ = [
    "ARXIV_PAPERS_CHUNKS_MAPPING",
    "HYBRID_RRF_PIPELINE",
    "OpenSearchClient",
    "QueryBuilder",
    "make_opensearch_client",
    "make_opensearch_client_fresh",
]
