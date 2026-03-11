"""Tests for the retrieval node (S6.3).

TDD: Write tests FIRST, then implement to make them pass.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from src.schemas.api.search import SearchHit
from src.services.agents.context import AgentContext
from src.services.agents.models import SourceItem
from src.services.agents.nodes.retrieve_node import (
    ainvoke_retrieve_step,
    convert_search_hits_to_sources,
)
from src.services.agents.state import AgentState, create_initial_state
from src.services.retrieval.pipeline import RetrievalResult

# ── Fixtures ──────────────────────────────────────────────────────────


def _make_search_hit(
    arxiv_id: str = "2301.00001",
    title: str = "Test Paper",
    authors: list[str] | None = None,
    score: float = 0.85,
    chunk_text: str = "Some chunk text",
    pdf_url: str = "",
) -> SearchHit:
    return SearchHit(
        arxiv_id=arxiv_id,
        title=title,
        authors=authors or ["Author A", "Author B"],
        score=score,
        chunk_text=chunk_text,
        pdf_url=pdf_url,
        chunk_id=f"chunk-{arxiv_id}",
    )


def _make_retrieval_result(
    hits: list[SearchHit] | None = None,
    stages: list[str] | None = None,
    total_candidates: int = 10,
    timings: dict[str, float] | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        results=hits or [],
        query="test query",
        stages_executed=stages or ["hybrid_search", "rerank"],
        total_candidates=total_candidates,
        timings=timings or {"hybrid_search": 0.12, "rerank": 0.05},
    )


def _make_context(pipeline: AsyncMock | None = None, top_k: int = 5) -> AgentContext:
    return AgentContext(
        llm_provider=MagicMock(),
        retrieval_pipeline=pipeline,
        top_k=top_k,
    )


def _make_state(query: str = "What is attention mechanism?", **overrides) -> AgentState:
    state = create_initial_state(query)
    for key, val in overrides.items():
        state[key] = val  # type: ignore[literal-required]
    return state


# ── FR-2: SearchHit → SourceItem Conversion ──────────────────────────


class TestConvertSearchHitsToSources:
    def test_search_hit_to_source_item_mapping(self) -> None:
        """Verify each field maps correctly including URL construction."""
        hit = _make_search_hit(
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            authors=["Vaswani", "Shazeer"],
            score=0.95,
            chunk_text="The dominant sequence transduction models...",
            pdf_url="https://arxiv.org/pdf/1706.03762",
        )

        sources = convert_search_hits_to_sources([hit])

        assert len(sources) == 1
        s = sources[0]
        assert isinstance(s, SourceItem)
        assert s.arxiv_id == "1706.03762"
        assert s.title == "Attention Is All You Need"
        assert s.authors == ["Vaswani", "Shazeer"]
        assert s.relevance_score == 0.95
        assert s.chunk_text == "The dominant sequence transduction models..."
        assert "1706.03762" in s.url

    def test_url_constructed_from_arxiv_id_when_no_pdf_url(self) -> None:
        """When pdf_url is empty, URL is constructed from arxiv_id."""
        hit = _make_search_hit(arxiv_id="2301.00001", pdf_url="")
        sources = convert_search_hits_to_sources([hit])
        assert sources[0].url == "https://arxiv.org/abs/2301.00001"

    def test_url_uses_pdf_url_when_provided(self) -> None:
        """When pdf_url is present, it's used for the URL."""
        hit = _make_search_hit(pdf_url="https://arxiv.org/pdf/2301.00001")
        sources = convert_search_hits_to_sources([hit])
        assert sources[0].url == "https://arxiv.org/pdf/2301.00001"

    def test_search_hit_missing_arxiv_id_skipped(self) -> None:
        """Hits with empty arxiv_id are filtered out."""
        hits = [
            _make_search_hit(arxiv_id="2301.00001"),
            _make_search_hit(arxiv_id=""),
            _make_search_hit(arxiv_id="2301.00003"),
        ]
        sources = convert_search_hits_to_sources(hits)
        assert len(sources) == 2
        assert sources[0].arxiv_id == "2301.00001"
        assert sources[1].arxiv_id == "2301.00003"

    def test_empty_hits_returns_empty_list(self) -> None:
        sources = convert_search_hits_to_sources([])
        assert sources == []

    def test_multiple_hits_preserve_order(self) -> None:
        hits = [
            _make_search_hit(arxiv_id="A", score=0.9),
            _make_search_hit(arxiv_id="B", score=0.8),
            _make_search_hit(arxiv_id="C", score=0.7),
        ]
        sources = convert_search_hits_to_sources(hits)
        assert [s.arxiv_id for s in sources] == ["A", "B", "C"]


# ── FR-1: Retrieve Documents via Pipeline ─────────────────────────────


class TestRetrieveStep:
    @pytest.mark.asyncio
    async def test_retrieve_step_returns_sources(self) -> None:
        """Given a mock pipeline returning 3 SearchHits, verify 3 SourceItems."""
        hits = [_make_search_hit(arxiv_id=f"230{i}.0000{i}") for i in range(1, 4)]
        pipeline = AsyncMock()
        pipeline.retrieve = AsyncMock(return_value=_make_retrieval_result(hits=hits))

        context = _make_context(pipeline=pipeline)
        state = _make_state()

        result = await ainvoke_retrieve_step(state, context)

        assert len(result["sources"]) == 3
        assert all(isinstance(s, SourceItem) for s in result["sources"])

    @pytest.mark.asyncio
    async def test_retrieve_step_uses_rewritten_query(self) -> None:
        """When rewritten_query is set, pipeline is called with it."""
        pipeline = AsyncMock()
        pipeline.retrieve = AsyncMock(return_value=_make_retrieval_result())
        context = _make_context(pipeline=pipeline)
        state = _make_state(rewritten_query="optimized query about transformers")

        await ainvoke_retrieve_step(state, context)

        pipeline.retrieve.assert_called_once()
        call_args = pipeline.retrieve.call_args
        used_query = call_args[0][0] if call_args[0] else call_args.kwargs.get("query", "")
        assert used_query == "optimized query about transformers"

    @pytest.mark.asyncio
    async def test_retrieve_step_falls_back_to_original_query(self) -> None:
        """When rewritten_query is None, uses original_query."""
        pipeline = AsyncMock()
        pipeline.retrieve = AsyncMock(return_value=_make_retrieval_result())
        context = _make_context(pipeline=pipeline)
        state = _make_state(query="What is attention mechanism?")

        await ainvoke_retrieve_step(state, context)

        pipeline.retrieve.assert_called_once()
        call_args = pipeline.retrieve.call_args
        # The query should be the original one
        used_query = call_args[0][0] if call_args[0] else call_args.kwargs.get("query", "")
        assert used_query == "What is attention mechanism?"

    @pytest.mark.asyncio
    async def test_retrieve_step_empty_results(self) -> None:
        """When pipeline returns zero hits, sources is empty list."""
        pipeline = AsyncMock()
        pipeline.retrieve = AsyncMock(return_value=_make_retrieval_result(hits=[]))
        context = _make_context(pipeline=pipeline)
        state = _make_state()

        result = await ainvoke_retrieve_step(state, context)

        assert result["sources"] == []


# ── FR-3: Retrieval Attempt Tracking ──────────────────────────────────


class TestRetrievalAttemptTracking:
    @pytest.mark.asyncio
    async def test_retrieve_step_increments_attempts(self) -> None:
        """Verify retrieval_attempts goes from 0 to 1."""
        pipeline = AsyncMock()
        pipeline.retrieve = AsyncMock(return_value=_make_retrieval_result())
        context = _make_context(pipeline=pipeline)
        state = _make_state()

        result = await ainvoke_retrieve_step(state, context)

        assert result["retrieval_attempts"] == 1

    @pytest.mark.asyncio
    async def test_retrieve_step_increments_from_existing(self) -> None:
        """Verify retrieval_attempts increments from current value."""
        pipeline = AsyncMock()
        pipeline.retrieve = AsyncMock(return_value=_make_retrieval_result())
        context = _make_context(pipeline=pipeline)
        state = _make_state(retrieval_attempts=2)

        result = await ainvoke_retrieve_step(state, context)

        assert result["retrieval_attempts"] == 3


# ── FR-1 Edge Cases: Pipeline None / Exception ───────────────────────


class TestRetrieveStepEdgeCases:
    @pytest.mark.asyncio
    async def test_retrieve_step_pipeline_none(self) -> None:
        """When retrieval_pipeline is None, returns empty sources with error metadata."""
        context = _make_context(pipeline=None)
        state = _make_state()

        result = await ainvoke_retrieve_step(state, context)

        assert result["sources"] == []
        assert "retrieval" in result["metadata"]
        assert "error" in result["metadata"]["retrieval"]

    @pytest.mark.asyncio
    async def test_retrieve_step_pipeline_exception(self) -> None:
        """When pipeline raises, returns empty sources, logs error."""
        pipeline = AsyncMock()
        pipeline.retrieve = AsyncMock(side_effect=RuntimeError("connection failed"))
        context = _make_context(pipeline=pipeline)
        state = _make_state()

        result = await ainvoke_retrieve_step(state, context)

        assert result["sources"] == []
        assert "retrieval" in result["metadata"]
        assert "error" in result["metadata"]["retrieval"]


# ── FR-4: Metadata Enrichment ─────────────────────────────────────────


class TestMetadataEnrichment:
    @pytest.mark.asyncio
    async def test_metadata_enrichment(self) -> None:
        """Verify stages_executed, timings, total_candidates are stored in metadata."""
        hits = [_make_search_hit()]
        pipeline = AsyncMock()
        pipeline.retrieve = AsyncMock(
            return_value=_make_retrieval_result(
                hits=hits,
                stages=["multi_query", "hybrid_search", "rerank"],
                total_candidates=25,
                timings={"multi_query": 0.3, "hybrid_search": 0.12, "rerank": 0.05},
            )
        )
        context = _make_context(pipeline=pipeline)
        state = _make_state()

        result = await ainvoke_retrieve_step(state, context)

        meta = result["metadata"]["retrieval"]
        assert meta["stages_executed"] == ["multi_query", "hybrid_search", "rerank"]
        assert meta["total_candidates"] == 25
        assert meta["timings"]["multi_query"] == 0.3
        assert meta["query_used"] == "What is attention mechanism?"
        assert meta["num_results"] == 1
