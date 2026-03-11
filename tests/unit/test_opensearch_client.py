"""Tests for OpenSearch client, index config, query builder, and factory.

TDD tests for Spec S4.1 -- OpenSearch Client + Index Configuration.
All OpenSearch API calls are mocked (no real cluster needed).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from src.services.opensearch.client import OpenSearchClient
from src.services.opensearch.factory import make_opensearch_client_fresh
from src.services.opensearch.index_config import (
    ARXIV_PAPERS_CHUNKS_MAPPING,
    HYBRID_RRF_PIPELINE,
)
from src.services.opensearch.query_builder import QueryBuilder

# ─── Index Config Tests ────────────────────────────────────────────


class TestIndexConfig:
    """Tests for index_config.py mappings and pipeline."""

    def test_chunks_mapping_has_settings(self):
        assert "settings" in ARXIV_PAPERS_CHUNKS_MAPPING
        settings = ARXIV_PAPERS_CHUNKS_MAPPING["settings"]
        assert settings["index.knn"] is True
        assert settings["number_of_shards"] == 1
        assert settings["number_of_replicas"] == 0

    def test_chunks_mapping_has_analyzers(self):
        analyzers = ARXIV_PAPERS_CHUNKS_MAPPING["settings"]["analysis"]["analyzer"]
        assert "text_analyzer" in analyzers
        assert "standard_analyzer" in analyzers
        assert analyzers["text_analyzer"]["type"] == "custom"
        assert "snowball" in analyzers["text_analyzer"]["filter"]

    def test_chunks_mapping_has_knn_vector_field(self):
        props = ARXIV_PAPERS_CHUNKS_MAPPING["mappings"]["properties"]
        assert "embedding" in props
        emb = props["embedding"]
        assert emb["type"] == "knn_vector"
        assert emb["dimension"] == 1024
        assert emb["method"]["name"] == "hnsw"
        assert emb["method"]["space_type"] == "cosinesimil"
        assert emb["method"]["engine"] == "nmslib"
        assert emb["method"]["parameters"]["ef_construction"] == 512
        assert emb["method"]["parameters"]["m"] == 16

    def test_chunks_mapping_has_required_fields(self):
        props = ARXIV_PAPERS_CHUNKS_MAPPING["mappings"]["properties"]
        required = [
            "chunk_id",
            "arxiv_id",
            "paper_id",
            "chunk_index",
            "chunk_text",
            "chunk_word_count",
            "start_char",
            "end_char",
            "embedding",
            "title",
            "authors",
            "abstract",
            "categories",
            "published_date",
            "section_title",
            "parent_chunk_id",
            "embedding_model",
            "created_at",
            "updated_at",
        ]
        for field in required:
            assert field in props, f"Missing field: {field}"

    def test_chunks_mapping_strict_dynamic(self):
        assert ARXIV_PAPERS_CHUNKS_MAPPING["mappings"]["dynamic"] == "strict"

    def test_rrf_pipeline_has_correct_structure(self):
        assert "id" in HYBRID_RRF_PIPELINE
        assert "phase_results_processors" in HYBRID_RRF_PIPELINE
        processor = HYBRID_RRF_PIPELINE["phase_results_processors"][0]
        assert "score-ranker-processor" in processor
        combo = processor["score-ranker-processor"]["combination"]
        assert combo["technique"] == "rrf"
        assert combo["rank_constant"] == 60


# ─── Query Builder Tests ───────────────────────────────────────────


class TestQueryBuilder:
    """Tests for QueryBuilder."""

    def test_basic_bm25_query(self):
        builder = QueryBuilder(query="transformers", size=10)
        body = builder.build()
        assert "query" in body
        assert body["size"] == 10
        # Default is chunk mode
        must = body["query"]["bool"]["must"]
        assert len(must) == 1
        mm = must[0]["multi_match"]
        assert mm["query"] == "transformers"
        assert mm["fuzziness"] == "AUTO"

    def test_chunk_mode_fields(self):
        builder = QueryBuilder(query="test", search_chunks=True)
        body = builder.build()
        mm = body["query"]["bool"]["must"][0]["multi_match"]
        assert "chunk_text^3" in mm["fields"]
        assert "title^2" in mm["fields"]
        assert "abstract^1" in mm["fields"]

    def test_paper_mode_fields(self):
        builder = QueryBuilder(query="test", search_chunks=False)
        body = builder.build()
        mm = body["query"]["bool"]["must"][0]["multi_match"]
        assert "title^3" in mm["fields"]
        assert "abstract^2" in mm["fields"]
        assert "authors^1" in mm["fields"]

    def test_with_categories_filter(self):
        builder = QueryBuilder(query="test", categories=["cs.AI", "cs.CL"])
        body = builder.build()
        filters = body["query"]["bool"]["filter"]
        assert len(filters) == 1
        assert filters[0] == {"terms": {"categories": ["cs.AI", "cs.CL"]}}

    def test_empty_query_uses_match_all(self):
        builder = QueryBuilder(query="", size=5)
        body = builder.build()
        must = body["query"]["bool"]["must"]
        assert must == [{"match_all": {}}]

    def test_latest_papers_sort(self):
        builder = QueryBuilder(query="test", latest_papers=True)
        body = builder.build()
        assert "sort" in body
        assert body["sort"][0] == {"published_date": {"order": "desc"}}

    def test_relevance_sort_no_sort_key(self):
        builder = QueryBuilder(query="test", latest_papers=False)
        body = builder.build()
        # No explicit sort → relevance scoring
        assert body.get("sort") is None

    def test_source_excludes_embedding_in_chunk_mode(self):
        builder = QueryBuilder(query="test", search_chunks=True)
        body = builder.build()
        assert body["_source"] == {"excludes": ["embedding"]}

    def test_source_fields_in_paper_mode(self):
        builder = QueryBuilder(query="test", search_chunks=False)
        body = builder.build()
        assert isinstance(body["_source"], list)
        assert "arxiv_id" in body["_source"]
        assert "title" in body["_source"]

    def test_highlight_config(self):
        builder = QueryBuilder(query="test", search_chunks=True)
        body = builder.build()
        assert "highlight" in body
        assert "chunk_text" in body["highlight"]["fields"]

    def test_pagination(self):
        builder = QueryBuilder(query="test", size=5, from_=10)
        body = builder.build()
        assert body["size"] == 5
        assert body["from"] == 10


# ─── OpenSearch Client Tests ──────────────────────────────────────


@pytest.fixture()
def mock_settings():
    """Create mock settings for OpenSearch client."""
    settings = MagicMock()
    settings.opensearch.host = "http://localhost:9201"
    settings.opensearch.index_name = "arxiv-papers"
    settings.opensearch.chunk_index_suffix = "chunks"
    settings.opensearch.vector_dimension = 1024
    settings.opensearch.vector_space_type = "cosinesimil"
    settings.opensearch.rrf_pipeline_name = "hybrid-rrf-pipeline"
    return settings


@pytest.fixture()
def mock_os_client(mock_settings):
    """Create OpenSearchClient with mocked opensearchpy.OpenSearch."""
    with patch("src.services.opensearch.client.OpenSearch") as mock_os_cls:
        mock_instance = MagicMock()
        mock_os_cls.return_value = mock_instance
        client = OpenSearchClient(host="http://localhost:9201", settings=mock_settings)
        # Expose the underlying mock for assertions
        client._mock_os = mock_instance
        yield client


class TestOpenSearchClientHealth:
    """Tests for health check."""

    def test_health_check_healthy(self, mock_os_client):
        mock_os_client._mock_os.cluster.health.return_value = {"status": "green"}
        assert mock_os_client.health_check() is True

    def test_health_check_yellow(self, mock_os_client):
        mock_os_client._mock_os.cluster.health.return_value = {"status": "yellow"}
        assert mock_os_client.health_check() is True

    def test_health_check_red(self, mock_os_client):
        mock_os_client._mock_os.cluster.health.return_value = {"status": "red"}
        assert mock_os_client.health_check() is False

    def test_health_check_exception(self, mock_os_client):
        mock_os_client._mock_os.cluster.health.side_effect = ConnectionError("refused")
        assert mock_os_client.health_check() is False


class TestOpenSearchClientSetup:
    """Tests for index and pipeline setup."""

    def test_setup_indices_calls_both(self, mock_os_client):
        mock_os_client._mock_os.indices.exists.return_value = False
        mock_os_client._mock_os.indices.create.return_value = {"acknowledged": True}
        # Mock pipeline check to raise (not found)
        mock_os_client._mock_os.ingest.get_pipeline.side_effect = Exception("not found")
        mock_os_client._mock_os.transport.perform_request.return_value = {"acknowledged": True}

        result = mock_os_client.setup_indices()
        assert "hybrid_index" in result
        assert "rrf_pipeline" in result

    def test_create_hybrid_index_new(self, mock_os_client):
        mock_os_client._mock_os.indices.exists.return_value = False
        mock_os_client._mock_os.indices.create.return_value = {"acknowledged": True}

        result = mock_os_client._create_hybrid_index()
        assert result is True
        mock_os_client._mock_os.indices.create.assert_called_once()

    def test_create_hybrid_index_exists(self, mock_os_client):
        mock_os_client._mock_os.indices.exists.return_value = True

        result = mock_os_client._create_hybrid_index()
        assert result is False
        mock_os_client._mock_os.indices.create.assert_not_called()

    def test_create_hybrid_index_force(self, mock_os_client):
        mock_os_client._mock_os.indices.exists.side_effect = [True, False]
        mock_os_client._mock_os.indices.delete.return_value = {"acknowledged": True}
        mock_os_client._mock_os.indices.create.return_value = {"acknowledged": True}

        result = mock_os_client._create_hybrid_index(force=True)
        assert result is True
        mock_os_client._mock_os.indices.delete.assert_called_once()
        mock_os_client._mock_os.indices.create.assert_called_once()


class TestOpenSearchClientSearch:
    """Tests for search methods."""

    def _make_search_response(self, hits_data: list[dict]) -> dict:
        """Helper to build mock search response."""
        return {
            "hits": {
                "total": {"value": len(hits_data)},
                "hits": [
                    {
                        "_id": f"chunk_{i}",
                        "_score": 1.0 - i * 0.1,
                        "_source": h,
                    }
                    for i, h in enumerate(hits_data)
                ],
            }
        }

    def test_search_bm25(self, mock_os_client):
        mock_os_client._mock_os.search.return_value = self._make_search_response(
            [{"chunk_text": "transformers are great", "arxiv_id": "2301.00001"}]
        )
        result = mock_os_client.search_papers(query="transformers", size=10)
        assert result["total"] == 1
        assert len(result["hits"]) == 1
        assert result["hits"][0]["score"] == 1.0
        assert result["hits"][0]["chunk_id"] == "chunk_0"

    def test_search_vectors(self, mock_os_client):
        embedding = [0.1] * 1024
        mock_os_client._mock_os.search.return_value = self._make_search_response(
            [{"chunk_text": "attention mechanism", "arxiv_id": "1706.03762"}]
        )
        result = mock_os_client.search_chunks_vectors(query_embedding=embedding, size=5)
        assert result["total"] == 1
        assert result["hits"][0]["arxiv_id"] == "1706.03762"

    def test_search_hybrid(self, mock_os_client):
        embedding = [0.1] * 1024
        mock_os_client._mock_os.search.return_value = self._make_search_response(
            [{"chunk_text": "hybrid result", "arxiv_id": "2301.00002"}]
        )
        result = mock_os_client.search_chunks_hybrid(query="transformers", query_embedding=embedding, size=5)
        assert result["total"] == 1
        # Verify search_pipeline param was used
        call_kwargs = mock_os_client._mock_os.search.call_args
        assert call_kwargs.kwargs.get("params", {}).get("search_pipeline") == "hybrid-rrf-pipeline"

    def test_search_unified_no_embedding_falls_back_bm25(self, mock_os_client):
        mock_os_client._mock_os.search.return_value = self._make_search_response(
            [{"chunk_text": "bm25 only", "arxiv_id": "2301.00003"}]
        )
        result = mock_os_client.search_unified(query="test", size=5)
        assert result["total"] == 1
        # Should NOT use search_pipeline (BM25 only)
        call_kwargs = mock_os_client._mock_os.search.call_args
        assert "params" not in (call_kwargs.kwargs or {})

    def test_search_unified_with_embedding_uses_hybrid(self, mock_os_client):
        embedding = [0.1] * 1024
        mock_os_client._mock_os.search.return_value = self._make_search_response(
            [{"chunk_text": "hybrid", "arxiv_id": "2301.00004"}]
        )
        result = mock_os_client.search_unified(query="test", query_embedding=embedding, size=5, use_hybrid=True)
        assert result["total"] == 1
        call_kwargs = mock_os_client._mock_os.search.call_args
        assert call_kwargs.kwargs.get("params", {}).get("search_pipeline") == "hybrid-rrf-pipeline"

    def test_search_unified_hybrid_disabled(self, mock_os_client):
        embedding = [0.1] * 1024
        mock_os_client._mock_os.search.return_value = self._make_search_response(
            [{"chunk_text": "bm25 forced", "arxiv_id": "2301.00005"}]
        )
        result = mock_os_client.search_unified(query="test", query_embedding=embedding, size=5, use_hybrid=False)
        assert result["total"] == 1

    def test_search_with_min_score_filtering(self, mock_os_client):
        mock_os_client._mock_os.search.return_value = {
            "hits": {
                "total": {"value": 2},
                "hits": [
                    {"_id": "c1", "_score": 0.9, "_source": {"chunk_text": "high"}},
                    {"_id": "c2", "_score": 0.1, "_source": {"chunk_text": "low"}},
                ],
            }
        }
        result = mock_os_client.search_unified(query="test", min_score=0.5)
        assert len(result["hits"]) == 1
        assert result["hits"][0]["chunk_text"] == "high"


class TestOpenSearchClientBulkAndLifecycle:
    """Tests for bulk indexing and chunk lifecycle."""

    def test_bulk_index_chunks(self, mock_os_client):
        chunks = [
            {"chunk_data": {"chunk_text": "hello", "arxiv_id": "123"}, "embedding": [0.1] * 1024},
            {"chunk_data": {"chunk_text": "world", "arxiv_id": "123"}, "embedding": [0.2] * 1024},
        ]
        with patch("src.services.opensearch.client.helpers") as mock_helpers:
            mock_helpers.bulk.return_value = (2, [])
            result = mock_os_client.bulk_index_chunks(chunks)
            assert result["success"] == 2
            assert result["failed"] == 0

    def test_delete_paper_chunks(self, mock_os_client):
        mock_os_client._mock_os.delete_by_query.return_value = {"deleted": 5}
        result = mock_os_client.delete_paper_chunks("2301.00001")
        assert result is True
        mock_os_client._mock_os.delete_by_query.assert_called_once()

    def test_delete_paper_chunks_none_found(self, mock_os_client):
        mock_os_client._mock_os.delete_by_query.return_value = {"deleted": 0}
        result = mock_os_client.delete_paper_chunks("nonexistent")
        assert result is False

    def test_get_chunks_by_paper(self, mock_os_client):
        mock_os_client._mock_os.search.return_value = {
            "hits": {
                "hits": [
                    {"_id": "c1", "_source": {"chunk_text": "first", "chunk_index": 0}},
                    {"_id": "c2", "_source": {"chunk_text": "second", "chunk_index": 1}},
                ]
            }
        }
        chunks = mock_os_client.get_chunks_by_paper("2301.00001")
        assert len(chunks) == 2
        assert chunks[0]["chunk_id"] == "c1"
        assert chunks[1]["chunk_id"] == "c2"

    def test_get_index_stats(self, mock_os_client):
        mock_os_client._mock_os.indices.exists.return_value = True
        mock_os_client._mock_os.indices.stats.return_value = {
            "indices": {
                "arxiv-papers-chunks": {
                    "total": {
                        "docs": {"count": 100, "deleted": 5},
                        "store": {"size_in_bytes": 1024000},
                    }
                }
            }
        }
        stats = mock_os_client.get_index_stats()
        assert stats["exists"] is True
        assert stats["document_count"] == 100
        assert stats["size_in_bytes"] == 1024000

    def test_get_index_stats_not_exists(self, mock_os_client):
        mock_os_client._mock_os.indices.exists.return_value = False
        stats = mock_os_client.get_index_stats()
        assert stats["exists"] is False
        assert stats["document_count"] == 0


# ─── Factory Tests ─────────────────────────────────────────────────


class TestFactory:
    """Tests for factory functions."""

    def test_fresh_instance_creates_new(self, mock_settings):
        with patch("src.services.opensearch.factory.OpenSearchClient") as mock_client_cls:
            mock_client_cls.return_value = MagicMock()
            make_opensearch_client_fresh(settings=mock_settings)
            make_opensearch_client_fresh(settings=mock_settings)
            assert mock_client_cls.call_count == 2

    def test_fresh_instance_host_override(self, mock_settings):
        with patch("src.services.opensearch.factory.OpenSearchClient") as mock_client_cls:
            mock_client_cls.return_value = MagicMock()
            make_opensearch_client_fresh(settings=mock_settings, host="http://custom:9200")
            mock_client_cls.assert_called_once_with(host="http://custom:9200", settings=mock_settings)
