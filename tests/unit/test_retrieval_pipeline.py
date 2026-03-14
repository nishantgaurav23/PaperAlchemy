"""Tests for the unified advanced retrieval pipeline (S4b.5).

TDD: Write all tests first, then implement to make them pass.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from src.config import RetrievalPipelineSettings
from src.schemas.api.search import SearchHit
from src.services.retrieval.hyde import HyDEResult
from src.services.retrieval.multi_query import MultiQueryResult

# ── Fixtures ──────────────────────────────────────────────────────────


def _make_hit(chunk_id: str, score: float = 0.5, **kwargs) -> SearchHit:
    """Helper to build SearchHit instances."""
    return SearchHit(
        chunk_id=chunk_id,
        score=score,
        arxiv_id=kwargs.get("arxiv_id", "2301.00001"),
        title=kwargs.get("title", "Test Paper"),
        authors=kwargs.get("authors", ["Author A"]),
        chunk_text=kwargs.get("chunk_text", f"Chunk text for {chunk_id}"),
    )


@pytest.fixture
def settings():
    return RetrievalPipelineSettings(
        multi_query_enabled=True,
        hyde_enabled=True,
        reranker_enabled=True,
        parent_expansion_enabled=True,
        retrieval_top_k=20,
        final_top_k=5,
    )


@pytest.fixture
def mock_multi_query():
    svc = AsyncMock()
    svc.retrieve_with_multi_query = AsyncMock(
        return_value=MultiQueryResult(
            original_query="test query",
            generated_queries=["variation 1", "variation 2"],
            results=[
                _make_hit("chunk_1", 0.9),
                _make_hit("chunk_2", 0.8),
                _make_hit("chunk_3", 0.7),
            ],
        )
    )
    return svc


@pytest.fixture
def mock_hyde():
    svc = AsyncMock()
    svc.retrieve_with_hyde = AsyncMock(
        return_value=HyDEResult(
            hypothetical_document="A hypothetical passage about test query",
            query_embedding=[0.1] * 1024,
            results=[
                _make_hit("chunk_2", 0.85),
                _make_hit("chunk_4", 0.75),
            ],
        )
    )
    return svc


@pytest.fixture
def mock_reranker():
    svc = AsyncMock()

    async def _rerank(query, hits, top_k=None):
        reranked = sorted(hits, key=lambda h: h.score, reverse=True)
        if top_k:
            reranked = reranked[:top_k]
        return [h.model_copy(update={"score": 0.95 - i * 0.1}) for i, h in enumerate(reranked)]

    svc.rerank_search_hits = AsyncMock(side_effect=_rerank)
    return svc


@pytest.fixture
def mock_parent_child():
    chunker = MagicMock()

    def _expand(child_results, os_client):
        return list(child_results)

    chunker.expand_to_parents = MagicMock(side_effect=_expand)
    return chunker


@pytest.fixture
def mock_opensearch():
    client = MagicMock()
    _hybrid_return = {
        "total": 2,
        "hits": [
            {
                "chunk_id": "chunk_fallback_1",
                "score": 0.6,
                "chunk_text": "Fallback text 1",
                "arxiv_id": "2301.00001",
                "title": "Fallback Paper",
                "authors": ["Author A"],
            },
            {
                "chunk_id": "chunk_fallback_2",
                "score": 0.5,
                "chunk_text": "Fallback text 2",
                "arxiv_id": "2301.00002",
                "title": "Fallback Paper 2",
                "authors": ["Author B"],
            },
        ],
    }
    client.search_chunks_hybrid = MagicMock(return_value=_hybrid_return)
    client.asearch_chunks_hybrid = AsyncMock(return_value=_hybrid_return)
    return client


@pytest.fixture
def mock_embeddings():
    client = AsyncMock()
    client.embed_query = AsyncMock(return_value=[0.1] * 1024)
    return client


@pytest.fixture
def pipeline(
    settings,
    mock_multi_query,
    mock_hyde,
    mock_reranker,
    mock_parent_child,
    mock_opensearch,
    mock_embeddings,
):
    from src.services.retrieval.pipeline import RetrievalPipeline

    return RetrievalPipeline(
        settings=settings,
        multi_query_service=mock_multi_query,
        hyde_service=mock_hyde,
        reranker_service=mock_reranker,
        parent_child_chunker=mock_parent_child,
        opensearch_client=mock_opensearch,
        embeddings_client=mock_embeddings,
    )


def _make_pipeline(
    settings,
    mock_multi_query,
    mock_hyde,
    mock_reranker,
    mock_parent_child,
    mock_opensearch,
    mock_embeddings,
):
    from src.services.retrieval.pipeline import RetrievalPipeline

    return RetrievalPipeline(
        settings=settings,
        multi_query_service=mock_multi_query,
        hyde_service=mock_hyde,
        reranker_service=mock_reranker,
        parent_child_chunker=mock_parent_child,
        opensearch_client=mock_opensearch,
        embeddings_client=mock_embeddings,
    )


# ── FR-1: Pipeline Initialization ────────────────────────────────────


class TestPipelineInit:
    def test_pipeline_init(self, pipeline):
        """Pipeline accepts all dependencies and stores them."""
        from src.services.retrieval.pipeline import RetrievalPipeline

        assert isinstance(pipeline, RetrievalPipeline)

    def test_pipeline_init_with_disabled_settings(
        self, mock_multi_query, mock_hyde, mock_reranker, mock_parent_child, mock_opensearch, mock_embeddings
    ):
        """Pipeline works with disabled stages."""
        settings = RetrievalPipelineSettings(
            multi_query_enabled=False,
            hyde_enabled=False,
            reranker_enabled=False,
            parent_expansion_enabled=False,
        )
        p = _make_pipeline(
            settings, mock_multi_query, mock_hyde, mock_reranker, mock_parent_child, mock_opensearch, mock_embeddings
        )
        from src.services.retrieval.pipeline import RetrievalPipeline

        assert isinstance(p, RetrievalPipeline)


# ── FR-2: Full Pipeline Retrieval ────────────────────────────────────


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_full_pipeline_all_stages(self, pipeline, mock_multi_query, mock_hyde, mock_reranker, mock_parent_child):
        """All stages execute in correct order with data flowing through."""
        result = await pipeline.retrieve("test query")

        mock_multi_query.retrieve_with_multi_query.assert_called_once()
        mock_hyde.retrieve_with_hyde.assert_called_once()
        mock_reranker.rerank_search_hits.assert_called_once()
        mock_parent_child.expand_to_parents.assert_called_once()

        assert result.query == "test query"
        assert len(result.results) <= 5
        assert result.expanded_queries == ["variation 1", "variation 2"]
        assert result.hypothetical_document == "A hypothetical passage about test query"
        assert "multi_query" in result.stages_executed
        assert "hyde" in result.stages_executed
        assert "hybrid_search" in result.stages_executed
        assert "rerank" in result.stages_executed
        assert "parent_expand" in result.stages_executed

    @pytest.mark.asyncio
    async def test_pipeline_top_k_respected(self, pipeline):
        """Final results count <= final_top_k."""
        result = await pipeline.retrieve("test query", top_k=3)
        assert len(result.results) <= 3


# ── FR-3: Graceful Degradation ───────────────────────────────────────


class TestGracefulDegradation:
    @pytest.mark.asyncio
    async def test_pipeline_multi_query_failure(self, pipeline, mock_multi_query):
        """When multi-query fails, fall back to plain hybrid search."""
        mock_multi_query.retrieve_with_multi_query = AsyncMock(side_effect=Exception("LLM down"))

        result = await pipeline.retrieve("test query")

        assert result is not None
        assert len(result.results) > 0
        assert "multi_query" not in result.stages_executed
        assert "hybrid_search" in result.stages_executed

    @pytest.mark.asyncio
    async def test_pipeline_hyde_failure(self, pipeline, mock_hyde):
        """When HyDE fails, pipeline continues without HyDE results."""
        mock_hyde.retrieve_with_hyde = AsyncMock(side_effect=Exception("Embedding API down"))

        result = await pipeline.retrieve("test query")

        assert result is not None
        assert len(result.results) > 0
        assert "hyde" not in result.stages_executed
        assert result.hypothetical_document == ""

    @pytest.mark.asyncio
    async def test_pipeline_reranker_failure(self, pipeline, mock_reranker):
        """When reranker fails, return un-reranked merged results."""
        mock_reranker.rerank_search_hits = AsyncMock(side_effect=Exception("Model load failed"))

        result = await pipeline.retrieve("test query")

        assert result is not None
        assert len(result.results) > 0
        assert "rerank" not in result.stages_executed

    @pytest.mark.asyncio
    async def test_pipeline_parent_expansion_failure(self, pipeline, mock_parent_child):
        """When parent expansion fails, return child chunks as-is."""
        mock_parent_child.expand_to_parents = MagicMock(side_effect=Exception("OS fetch error"))

        result = await pipeline.retrieve("test query")

        assert result is not None
        assert len(result.results) > 0
        assert "parent_expand" not in result.stages_executed


# ── FR-4: Pipeline Configuration (Disabled Stages) ───────────────────


class TestDisabledStages:
    @pytest.mark.asyncio
    async def test_pipeline_multi_query_disabled(
        self, mock_multi_query, mock_hyde, mock_reranker, mock_parent_child, mock_opensearch, mock_embeddings
    ):
        """Multi-query disabled -> uses original query only."""
        settings = RetrievalPipelineSettings(multi_query_enabled=False)
        p = _make_pipeline(
            settings, mock_multi_query, mock_hyde, mock_reranker, mock_parent_child, mock_opensearch, mock_embeddings
        )
        result = await p.retrieve("test query")

        mock_multi_query.retrieve_with_multi_query.assert_not_called()
        assert "multi_query" not in result.stages_executed
        assert result.expanded_queries == []

    @pytest.mark.asyncio
    async def test_pipeline_hyde_disabled(
        self, mock_multi_query, mock_hyde, mock_reranker, mock_parent_child, mock_opensearch, mock_embeddings
    ):
        """HyDE disabled -> skips HyDE stage entirely."""
        settings = RetrievalPipelineSettings(hyde_enabled=False)
        p = _make_pipeline(
            settings, mock_multi_query, mock_hyde, mock_reranker, mock_parent_child, mock_opensearch, mock_embeddings
        )
        result = await p.retrieve("test query")

        mock_hyde.retrieve_with_hyde.assert_not_called()
        assert "hyde" not in result.stages_executed
        assert result.hypothetical_document == ""

    @pytest.mark.asyncio
    async def test_pipeline_reranker_disabled(
        self, mock_multi_query, mock_hyde, mock_reranker, mock_parent_child, mock_opensearch, mock_embeddings
    ):
        """Reranker disabled -> returns merged results sorted by score."""
        settings = RetrievalPipelineSettings(reranker_enabled=False)
        p = _make_pipeline(
            settings, mock_multi_query, mock_hyde, mock_reranker, mock_parent_child, mock_opensearch, mock_embeddings
        )
        result = await p.retrieve("test query")

        mock_reranker.rerank_search_hits.assert_not_called()
        assert "rerank" not in result.stages_executed

    @pytest.mark.asyncio
    async def test_pipeline_parent_expansion_disabled(
        self, mock_multi_query, mock_hyde, mock_reranker, mock_parent_child, mock_opensearch, mock_embeddings
    ):
        """Parent expansion disabled -> returns child chunks as-is."""
        settings = RetrievalPipelineSettings(parent_expansion_enabled=False)
        p = _make_pipeline(
            settings, mock_multi_query, mock_hyde, mock_reranker, mock_parent_child, mock_opensearch, mock_embeddings
        )
        result = await p.retrieve("test query")

        mock_parent_child.expand_to_parents.assert_not_called()
        assert "parent_expand" not in result.stages_executed

    @pytest.mark.asyncio
    async def test_pipeline_all_disabled(
        self, mock_multi_query, mock_hyde, mock_reranker, mock_parent_child, mock_opensearch, mock_embeddings
    ):
        """All advanced stages disabled -> falls back to plain hybrid search."""
        settings = RetrievalPipelineSettings(
            multi_query_enabled=False,
            hyde_enabled=False,
            reranker_enabled=False,
            parent_expansion_enabled=False,
        )
        p = _make_pipeline(
            settings, mock_multi_query, mock_hyde, mock_reranker, mock_parent_child, mock_opensearch, mock_embeddings
        )
        result = await p.retrieve("test query")

        mock_multi_query.retrieve_with_multi_query.assert_not_called()
        mock_hyde.retrieve_with_hyde.assert_not_called()
        mock_reranker.rerank_search_hits.assert_not_called()
        mock_parent_child.expand_to_parents.assert_not_called()

        assert "hybrid_search" in result.stages_executed
        assert len(result.results) > 0


# ── FR-5: Result Model & Metadata ────────────────────────────────────


class TestResultMetadata:
    @pytest.mark.asyncio
    async def test_pipeline_deduplication(self, pipeline):
        """Merging multi-query + HyDE deduplicates by chunk_id, keeps highest score."""
        result = await pipeline.retrieve("test query")

        chunk_ids = [h.chunk_id for h in result.results]
        assert len(chunk_ids) == len(set(chunk_ids))

    @pytest.mark.asyncio
    async def test_pipeline_timing_metadata(self, pipeline):
        """Timings dict populated for each executed stage."""
        result = await pipeline.retrieve("test query")

        assert isinstance(result.timings, dict)
        assert len(result.timings) > 0
        for _stage, duration in result.timings.items():
            assert isinstance(duration, float)
            assert duration >= 0.0

    @pytest.mark.asyncio
    async def test_pipeline_stages_executed_tracking(self, pipeline):
        """stages_executed list reflects actual execution."""
        result = await pipeline.retrieve("test query")

        assert isinstance(result.stages_executed, list)
        assert "multi_query" in result.stages_executed
        assert "hyde" in result.stages_executed
        assert "hybrid_search" in result.stages_executed
        assert "rerank" in result.stages_executed
        assert "parent_expand" in result.stages_executed

    @pytest.mark.asyncio
    async def test_pipeline_total_candidates_tracked(self, pipeline):
        """total_candidates reflects pre-rerank candidate count."""
        result = await pipeline.retrieve("test query")

        assert result.total_candidates > 0


# ── FR-6: Factory Function ───────────────────────────────────────────


class TestFactory:
    def test_factory_function(
        self, settings, mock_multi_query, mock_hyde, mock_reranker, mock_parent_child, mock_opensearch, mock_embeddings
    ):
        """Factory creates pipeline with all services."""
        from src.services.retrieval.factory import create_retrieval_pipeline

        p = create_retrieval_pipeline(
            settings=settings,
            multi_query_service=mock_multi_query,
            hyde_service=mock_hyde,
            reranker_service=mock_reranker,
            parent_child_chunker=mock_parent_child,
            opensearch_client=mock_opensearch,
            embeddings_client=mock_embeddings,
        )

        from src.services.retrieval.pipeline import RetrievalPipeline

        assert isinstance(p, RetrievalPipeline)

    def test_pipeline_settings_defaults(self):
        """RetrievalPipelineSettings has sensible defaults."""
        s = RetrievalPipelineSettings()
        assert s.multi_query_enabled is True
        assert s.hyde_enabled is True
        assert s.reranker_enabled is True
        assert s.parent_expansion_enabled is True
        assert s.retrieval_top_k == 20
        assert s.final_top_k == 5

    def test_pipeline_settings_in_root_settings(self):
        """RetrievalPipelineSettings is accessible from root Settings."""
        from src.config import Settings

        s = Settings()
        assert hasattr(s, "retrieval_pipeline")
        assert isinstance(s.retrieval_pipeline, RetrievalPipelineSettings)

    def test_pipeline_settings_from_env(self, monkeypatch):
        """Settings can be configured via environment variables."""
        monkeypatch.setenv("RETRIEVAL_PIPELINE__MULTI_QUERY_ENABLED", "false")
        monkeypatch.setenv("RETRIEVAL_PIPELINE__FINAL_TOP_K", "10")
        s = RetrievalPipelineSettings()
        assert s.multi_query_enabled is False
        assert s.final_top_k == 10
