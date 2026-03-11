"""OpenSearch client supporting BM25, vector, and hybrid search with native RRF.

Wraps opensearch-py with PaperAlchemy-specific logic: index management,
query building, result formatting, and chunk lifecycle.
"""

from __future__ import annotations

import logging
from typing import Any

from opensearchpy import OpenSearch, helpers
from src.config import Settings

from .index_config import ARXIV_PAPERS_CHUNKS_MAPPING, HYBRID_RRF_PIPELINE
from .query_builder import QueryBuilder

logger = logging.getLogger(__name__)


class OpenSearchClient:
    """OpenSearch client for BM25 and hybrid search with native RRF."""

    def __init__(self, host: str, settings: Settings):
        self.host = host
        self.settings = settings
        self.index_name = f"{settings.opensearch.index_name}-{settings.opensearch.chunk_index_suffix}"

        self.client = OpenSearch(
            hosts=[host],
            use_ssl=False,
            verify_certs=False,
            ssl_show_warn=False,
        )
        logger.info("OpenSearch client initialized with host: %s", host)

    # ── Health ──────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """Check if OpenSearch cluster is healthy (green or yellow)."""
        try:
            health = self.client.cluster.health()
            return health["status"] in ("green", "yellow")
        except Exception as e:
            logger.error("Health check failed: %s", e)
            return False

    # ── Index & Pipeline Setup ──────────────────────────────────────

    def setup_indices(self, force: bool = False) -> dict[str, bool]:
        """Setup the hybrid search index and RRF pipeline."""
        return {
            "hybrid_index": self._create_hybrid_index(force),
            "rrf_pipeline": self._create_rrf_pipeline(force),
        }

    def _create_hybrid_index(self, force: bool = False) -> bool:
        """Create hybrid index. Returns True if created, False if already exists."""
        try:
            if force and self.client.indices.exists(index=self.index_name):
                self.client.indices.delete(index=self.index_name)
                logger.info("Deleted existing hybrid index: %s", self.index_name)

            if not self.client.indices.exists(index=self.index_name):
                self.client.indices.create(index=self.index_name, body=ARXIV_PAPERS_CHUNKS_MAPPING)
                logger.info("Created hybrid index: %s", self.index_name)
                return True

            logger.info("Hybrid index already exists: %s", self.index_name)
            return False
        except Exception as e:
            logger.error("Error creating hybrid index: %s", e)
            raise

    def _create_rrf_pipeline(self, force: bool = False) -> bool:
        """Create RRF search pipeline. Returns True if created."""
        try:
            pipeline_id = HYBRID_RRF_PIPELINE["id"]

            if force:
                try:
                    self.client.ingest.get_pipeline(id=pipeline_id)
                    self.client.ingest.delete_pipeline(id=pipeline_id)
                    logger.info("Deleted existing RRF pipeline: %s", pipeline_id)
                except Exception:
                    pass

            try:
                self.client.ingest.get_pipeline(id=pipeline_id)
                logger.info("RRF pipeline already exists: %s", pipeline_id)
                return False
            except Exception:
                pass

            pipeline_body = {
                "description": HYBRID_RRF_PIPELINE["description"],
                "phase_results_processors": HYBRID_RRF_PIPELINE["phase_results_processors"],
            }
            self.client.transport.perform_request("PUT", f"/_search/pipeline/{pipeline_id}", body=pipeline_body)
            logger.info("Created RRF search pipeline: %s", pipeline_id)
            return True
        except Exception as e:
            logger.error("Error creating RRF pipeline: %s", e)
            raise

    # ── Search Methods ──────────────────────────────────────────────

    def search_papers(
        self,
        query: str,
        size: int = 10,
        from_: int = 0,
        categories: list[str] | None = None,
        latest: bool = True,
    ) -> dict[str, Any]:
        """BM25 keyword search for papers/chunks."""
        return self._search_bm25_only(query=query, size=size, from_=from_, categories=categories, latest=latest)

    def search_chunks_vectors(
        self,
        query_embedding: list[float],
        size: int = 10,
        categories: list[str] | None = None,
    ) -> dict[str, Any]:
        """Pure KNN vector search on chunk embeddings."""
        try:
            search_body: dict[str, Any] = {
                "size": size,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": query_embedding,
                            "k": size,
                        }
                    }
                },
                "_source": {"excludes": ["embedding"]},
            }

            if categories:
                search_body["query"] = {
                    "bool": {
                        "must": [search_body["query"]],
                        "filter": [{"terms": {"categories": categories}}],
                    }
                }

            response = self.client.search(index=self.index_name, body=search_body)
            return self._format_results(response)
        except Exception as e:
            logger.error("Vector search error: %s", e)
            return {"total": 0, "hits": []}

    def search_unified(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        size: int = 10,
        from_: int = 0,
        categories: list[str] | None = None,
        latest: bool = False,
        use_hybrid: bool = True,
        min_score: float = 0.0,
    ) -> dict[str, Any]:
        """Unified search: routes to BM25 or hybrid based on inputs."""
        try:
            if not query_embedding or not use_hybrid:
                return self._search_bm25_only(
                    query=query, size=size, from_=from_, categories=categories, latest=latest, min_score=min_score
                )
            return self._search_hybrid_native(
                query=query, query_embedding=query_embedding, size=size, categories=categories, min_score=min_score
            )
        except Exception as e:
            logger.error("Unified search error: %s", e)
            return {"total": 0, "hits": []}

    def search_chunks_hybrid(
        self,
        query: str,
        query_embedding: list[float],
        size: int = 10,
        categories: list[str] | None = None,
        min_score: float = 0.0,
    ) -> dict[str, Any]:
        """Hybrid search combining BM25 and vector via native RRF."""
        return self._search_hybrid_native(
            query=query, query_embedding=query_embedding, size=size, categories=categories, min_score=min_score
        )

    # ── Private Search Implementations ──────────────────────────────

    def _search_bm25_only(
        self,
        query: str,
        size: int,
        from_: int = 0,
        categories: list[str] | None = None,
        latest: bool = False,
        min_score: float = 0.0,
    ) -> dict[str, Any]:
        """Pure BM25 search implementation."""
        builder = QueryBuilder(
            query=query,
            size=size,
            from_=from_,
            categories=categories,
            latest_papers=latest,
            search_chunks=True,
        )
        search_body = builder.build()
        response = self.client.search(index=self.index_name, body=search_body)
        return self._format_results(response, min_score=min_score)

    def _search_hybrid_native(
        self,
        query: str,
        query_embedding: list[float],
        size: int,
        categories: list[str] | None = None,
        min_score: float = 0.0,
    ) -> dict[str, Any]:
        """Native OpenSearch hybrid search with RRF pipeline."""
        builder = QueryBuilder(
            query=query, size=size * 2, from_=0, categories=categories, latest_papers=False, search_chunks=True
        )
        bm25_search_body = builder.build()
        bm25_query = bm25_search_body["query"]

        hybrid_query = {
            "hybrid": {
                "queries": [
                    bm25_query,
                    {"knn": {"embedding": {"vector": query_embedding, "k": size * 2}}},
                ]
            }
        }

        search_body = {
            "size": size,
            "query": hybrid_query,
            "_source": bm25_search_body["_source"],
            "highlight": bm25_search_body["highlight"],
        }

        response = self.client.search(
            index=self.index_name, body=search_body, params={"search_pipeline": HYBRID_RRF_PIPELINE["id"]}
        )
        result = self._format_results(response, min_score=min_score)
        result["total"] = len(result["hits"])
        return result

    # ── Result Formatting ───────────────────────────────────────────

    @staticmethod
    def _format_results(response: dict[str, Any], min_score: float = 0.0) -> dict[str, Any]:
        """Format OpenSearch response into a standardized result dict."""
        results: dict[str, Any] = {"total": response["hits"]["total"]["value"], "hits": []}
        for hit in response["hits"]["hits"]:
            if hit["_score"] < min_score:
                continue
            chunk = hit["_source"]
            chunk["score"] = hit["_score"]
            chunk["chunk_id"] = hit["_id"]
            if "highlight" in hit:
                chunk["highlights"] = hit["highlight"]
            results["hits"].append(chunk)
        return results

    # ── Bulk Indexing ───────────────────────────────────────────────

    def bulk_index_chunks(self, chunks: list[dict[str, Any]]) -> dict[str, int]:
        """Bulk index multiple chunks with embeddings."""
        try:
            actions = []
            for chunk in chunks:
                chunk_data = chunk["chunk_data"].copy()
                chunk_data["embedding"] = chunk["embedding"]
                actions.append({"_index": self.index_name, "_source": chunk_data})

            success, failed = helpers.bulk(self.client, actions, refresh=True)
            logger.info("Bulk indexed %d chunks, %d failed", success, len(failed))
            return {"success": success, "failed": len(failed)}
        except Exception as e:
            logger.error("Bulk chunk indexing error: %s", e)
            raise

    # ── Chunk Lifecycle ─────────────────────────────────────────────

    def delete_paper_chunks(self, arxiv_id: str) -> bool:
        """Delete all chunks for a specific paper."""
        try:
            response = self.client.delete_by_query(
                index=self.index_name,
                body={"query": {"term": {"arxiv_id": arxiv_id}}},
                refresh=True,
            )
            deleted = response.get("deleted", 0)
            logger.info("Deleted %d chunks for paper %s", deleted, arxiv_id)
            return deleted > 0
        except Exception as e:
            logger.error("Error deleting chunks: %s", e)
            return False

    def get_chunks_by_paper(self, arxiv_id: str) -> list[dict[str, Any]]:
        """Get all chunks for a paper, sorted by chunk_index."""
        try:
            search_body = {
                "query": {"term": {"arxiv_id": arxiv_id}},
                "size": 1000,
                "sort": [{"chunk_index": "asc"}],
                "_source": {"excludes": ["embedding"]},
            }
            response = self.client.search(index=self.index_name, body=search_body)
            chunks = []
            for hit in response["hits"]["hits"]:
                chunk = hit["_source"]
                chunk["chunk_id"] = hit["_id"]
                chunks.append(chunk)
            return chunks
        except Exception as e:
            logger.error("Error getting chunks: %s", e)
            return []

    # ── Index Stats ─────────────────────────────────────────────────

    def get_index_stats(self) -> dict[str, Any]:
        """Get statistics for the chunk index."""
        try:
            if not self.client.indices.exists(index=self.index_name):
                return {"index_name": self.index_name, "exists": False, "document_count": 0}

            stats_response = self.client.indices.stats(index=self.index_name)
            index_stats = stats_response["indices"][self.index_name]["total"]
            return {
                "index_name": self.index_name,
                "exists": True,
                "document_count": index_stats["docs"]["count"],
                "deleted_count": index_stats["docs"]["deleted"],
                "size_in_bytes": index_stats["store"]["size_in_bytes"],
            }
        except Exception as e:
            logger.error("Error getting index stats: %s", e)
            return {"index_name": self.index_name, "exists": False, "document_count": 0, "error": str(e)}
