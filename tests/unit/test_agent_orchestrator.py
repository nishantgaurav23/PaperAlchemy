"""Tests for the agent orchestrator (S6.7).

Tests cover:
- Graph compilation and node registration
- Fast path: parallel KB + web → single LLM call
- No sources: KB empty + web empty → graceful fallback
- KB sources preferred over web sources
- Empty query validation
- Result extraction helpers
- Factory function
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from src.services.agents.context import AgentContext
from src.services.agents.models import GradingResult, GuardrailScoring, SourceItem
from src.services.rag.models import SourceReference

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider with get_langchain_model."""
    provider = MagicMock()
    mock_model = MagicMock()
    mock_structured = AsyncMock()
    mock_model.with_structured_output.return_value = mock_structured
    provider.get_langchain_model.return_value = mock_model
    return provider


@pytest.fixture
def mock_retrieval_pipeline():
    """Create a mock retrieval pipeline."""
    pipeline = AsyncMock()
    return pipeline


@pytest.fixture
def context(mock_llm_provider, mock_retrieval_pipeline):
    """Create an AgentContext with mocked services."""
    return AgentContext(
        llm_provider=mock_llm_provider,
        retrieval_pipeline=mock_retrieval_pipeline,
        model_name="test-model",
        guardrail_threshold=40,
        max_retrieval_attempts=3,
        top_k=5,
    )


# ---------------------------------------------------------------------------
# Test: Graph Compilation (FR-1)
# ---------------------------------------------------------------------------


class TestGraphCompilation:
    """FR-1: StateGraph construction and compilation."""

    def test_graph_compilation(self, mock_llm_provider, mock_retrieval_pipeline):
        """Graph compiles without error and has expected node names."""
        from src.services.agents.agentic_rag import AgenticRAGService

        service = AgenticRAGService(
            llm_provider=mock_llm_provider,
            retrieval_pipeline=mock_retrieval_pipeline,
        )

        assert service._compiled_graph is not None

        # Graph should have all expected nodes
        graph = service._compiled_graph.get_graph()
        node_ids = set(graph.nodes)
        assert "guardrail" in node_ids
        assert "out_of_scope" in node_ids
        assert "retrieve" in node_ids
        assert "grade_documents" in node_ids
        assert "rewrite_query" in node_ids
        assert "generate_answer" in node_ids


# ---------------------------------------------------------------------------
# Test: Fast Path — KB sources available
# ---------------------------------------------------------------------------


class TestFastPathWithKB:
    """Fast path: KB retrieval returns results → single LLM call."""

    @pytest.mark.asyncio
    async def test_kb_sources_used(self, mock_llm_provider, mock_retrieval_pipeline):
        """When KB has results, they are used for generation."""
        from src.schemas.api.search import SearchHit
        from src.services.agents.agentic_rag import AgenticRAGService

        mock_retrieval_pipeline.retrieve = AsyncMock(
            return_value=MagicMock(
                results=[
                    SearchHit(
                        arxiv_id="1706.03762",
                        title="Attention Is All You Need",
                        authors=["Vaswani et al."],
                        score=0.95,
                        chunk_text="The dominant sequence transduction models...",
                        pdf_url="",
                    ),
                ],
                stages_executed=["hybrid_search"],
                total_candidates=20,
                timings={"hybrid_search": 0.1},
            )
        )

        answer_text = "Transformers use self-attention [1]."
        model_mock = MagicMock()
        model_mock.ainvoke = AsyncMock(return_value=AIMessage(content=answer_text))
        mock_llm_provider.get_langchain_model.return_value = model_mock

        service = AgenticRAGService(
            llm_provider=mock_llm_provider,
            retrieval_pipeline=mock_retrieval_pipeline,
        )

        result = await service.ask("What are transformers?")

        assert result.answer is not None
        assert len(result.answer) > 0
        assert result.metadata["kb_sources"] == 1
        assert isinstance(result.sources, list)
        assert len(result.sources) == 1
        assert result.sources[0].arxiv_id == "1706.03762"


# ---------------------------------------------------------------------------
# Test: Fast Path — KB empty, web fallback
# ---------------------------------------------------------------------------


class TestFastPathWebFallback:
    """Fast path: KB empty → web search provides sources."""

    @pytest.mark.asyncio
    async def test_web_fallback_used(self, mock_llm_provider, mock_retrieval_pipeline):
        """When KB is empty but web search has results, web sources are used."""
        from src.services.agents.agentic_rag import AgenticRAGService
        from src.services.web_search.service import WebSearchResponse, WebSearchResult, WebSearchService

        # KB returns nothing
        mock_retrieval_pipeline.retrieve = AsyncMock(
            return_value=MagicMock(results=[], stages_executed=[], total_candidates=0, timings={})
        )

        # Web search returns results
        mock_web = AsyncMock(spec=WebSearchService)
        mock_web.search = AsyncMock(
            return_value=WebSearchResponse(
                query="test",
                results=[
                    WebSearchResult(title="Web Result", url="https://example.com/paper", snippet="Relevant content here.", source="example.com"),
                ],
            )
        )

        answer_text = "Based on web sources, the answer is X [1]."
        model_mock = MagicMock()
        model_mock.ainvoke = AsyncMock(return_value=AIMessage(content=answer_text))
        mock_llm_provider.get_langchain_model.return_value = model_mock

        service = AgenticRAGService(
            llm_provider=mock_llm_provider,
            retrieval_pipeline=mock_retrieval_pipeline,
            web_search_service=mock_web,
        )

        result = await service.ask("What is quantum computing?")

        assert result.answer is not None
        assert len(result.answer) > 0
        assert result.metadata["kb_sources"] == 0
        assert result.metadata["web_sources"] > 0


# ---------------------------------------------------------------------------
# Test: No sources at all
# ---------------------------------------------------------------------------


class TestNoSourcesAvailable:
    """Both KB and web return nothing."""

    @pytest.mark.asyncio
    async def test_no_sources_graceful(self, mock_llm_provider, mock_retrieval_pipeline):
        """When no sources are found anywhere, return a graceful message (no LLM call)."""
        from src.services.agents.agentic_rag import AgenticRAGService

        # KB returns nothing
        mock_retrieval_pipeline.retrieve = AsyncMock(
            return_value=MagicMock(results=[], stages_executed=[], total_candidates=0, timings={})
        )

        service = AgenticRAGService(
            llm_provider=mock_llm_provider,
            retrieval_pipeline=mock_retrieval_pipeline,
            # No web search service
        )

        result = await service.ask("Some obscure topic?")

        assert "couldn't find" in result.answer.lower() or "no relevant" in result.answer.lower()
        assert len(result.sources) == 0
        # No LLM call should have been made
        mock_llm_provider.get_langchain_model.assert_not_called()


# ---------------------------------------------------------------------------
# Test: KB preferred over web
# ---------------------------------------------------------------------------


class TestKBPreferredOverWeb:
    """When KB has sources, web sources are not used."""

    @pytest.mark.asyncio
    async def test_kb_preferred(self, mock_llm_provider, mock_retrieval_pipeline):
        """KB sources take priority even when web also returns results."""
        from src.schemas.api.search import SearchHit
        from src.services.agents.agentic_rag import AgenticRAGService
        from src.services.web_search.service import WebSearchResponse, WebSearchResult, WebSearchService

        mock_retrieval_pipeline.retrieve = AsyncMock(
            return_value=MagicMock(
                results=[
                    SearchHit(
                        arxiv_id="2301.00001",
                        title="KB Paper",
                        authors=["Author"],
                        score=0.9,
                        chunk_text="KB content.",
                        pdf_url="",
                    ),
                ],
                stages_executed=["hybrid_search"],
                total_candidates=5,
                timings={},
            )
        )

        mock_web = AsyncMock(spec=WebSearchService)
        mock_web.search = AsyncMock(
            return_value=WebSearchResponse(
                query="test",
                results=[WebSearchResult(title="Web", url="https://example.com", snippet="Web content.", source="example.com")],
            )
        )

        answer_text = "Answer from KB [1]."
        model_mock = MagicMock()
        model_mock.ainvoke = AsyncMock(return_value=AIMessage(content=answer_text))
        mock_llm_provider.get_langchain_model.return_value = model_mock

        service = AgenticRAGService(
            llm_provider=mock_llm_provider,
            retrieval_pipeline=mock_retrieval_pipeline,
            web_search_service=mock_web,
        )

        result = await service.ask("What is deep learning?")

        assert result.metadata["kb_sources"] == 1
        # KB sources used for generation
        assert result.sources[0].arxiv_id == "2301.00001"


# ---------------------------------------------------------------------------
# Test: Empty Query (FR-2 edge)
# ---------------------------------------------------------------------------


class TestEmptyQuery:
    """Empty query raises ValueError."""

    @pytest.mark.asyncio
    async def test_empty_query_raises(self, mock_llm_provider, mock_retrieval_pipeline):
        from src.services.agents.agentic_rag import AgenticRAGService

        service = AgenticRAGService(
            llm_provider=mock_llm_provider,
            retrieval_pipeline=mock_retrieval_pipeline,
        )

        with pytest.raises(ValueError, match="non-empty"):
            await service.ask("")

    @pytest.mark.asyncio
    async def test_whitespace_query_raises(self, mock_llm_provider, mock_retrieval_pipeline):
        from src.services.agents.agentic_rag import AgenticRAGService

        service = AgenticRAGService(
            llm_provider=mock_llm_provider,
            retrieval_pipeline=mock_retrieval_pipeline,
        )

        with pytest.raises(ValueError, match="non-empty"):
            await service.ask("   ")


# ---------------------------------------------------------------------------
# Test: Extract Answer (FR-4)
# ---------------------------------------------------------------------------


class TestExtractAnswer:
    """_extract_answer helper."""

    def test_extract_answer_from_ai_message(self):
        from src.services.agents.agentic_rag import AgenticRAGService

        state = {
            "messages": [
                HumanMessage(content="What is X?"),
                AIMessage(content="X is a thing [1]."),
            ],
        }

        answer = AgenticRAGService._extract_answer(state)
        assert answer == "X is a thing [1]."

    def test_extract_answer_empty_messages(self):
        from src.services.agents.agentic_rag import AgenticRAGService

        state = {"messages": []}

        answer = AgenticRAGService._extract_answer(state)
        assert "error" in answer.lower() or "unable" in answer.lower() or len(answer) > 0

    def test_extract_answer_no_ai_message(self):
        from src.services.agents.agentic_rag import AgenticRAGService

        state = {"messages": [HumanMessage(content="hello")]}

        answer = AgenticRAGService._extract_answer(state)
        assert len(answer) > 0  # Should return some fallback


# ---------------------------------------------------------------------------
# Test: Extract Sources (FR-4)
# ---------------------------------------------------------------------------


class TestExtractSources:
    """_extract_sources helper."""

    def test_extract_sources_converts_source_items(self):
        from src.services.agents.agentic_rag import AgenticRAGService

        state = {
            "relevant_sources": [
                SourceItem(
                    arxiv_id="1706.03762",
                    title="Attention Is All You Need",
                    authors=["Vaswani"],
                    url="https://arxiv.org/abs/1706.03762",
                    relevance_score=0.95,
                    chunk_text="Self-attention...",
                ),
                SourceItem(
                    arxiv_id="1810.04805",
                    title="BERT",
                    authors=["Devlin"],
                    url="https://arxiv.org/abs/1810.04805",
                    relevance_score=0.85,
                    chunk_text="Pre-training...",
                ),
            ],
        }

        sources = AgenticRAGService._extract_sources(state)

        assert len(sources) == 2
        assert isinstance(sources[0], SourceReference)
        assert sources[0].index == 1
        assert sources[0].arxiv_id == "1706.03762"
        assert sources[0].title == "Attention Is All You Need"
        assert sources[0].arxiv_url == "https://arxiv.org/abs/1706.03762"
        assert sources[1].index == 2

    def test_extract_sources_empty(self):
        from src.services.agents.agentic_rag import AgenticRAGService

        state = {"relevant_sources": []}
        sources = AgenticRAGService._extract_sources(state)
        assert sources == []

    def test_extract_sources_missing_field(self):
        from src.services.agents.agentic_rag import AgenticRAGService

        state = {}
        sources = AgenticRAGService._extract_sources(state)
        assert sources == []


# ---------------------------------------------------------------------------
# Test: Extract Reasoning Steps (FR-4)
# ---------------------------------------------------------------------------


class TestExtractReasoningSteps:
    """_extract_reasoning_steps helper."""

    def test_extract_reasoning_steps_full(self):
        from unittest.mock import MagicMock

        from src.services.agents.agentic_rag import AgenticRAGService

        state = {
            "guardrail_result": GuardrailScoring(score=85, reason="Research question"),
            "sources": [
                SourceItem(arxiv_id="1", title="P1", authors=[], url="u1", chunk_text="c1"),
                SourceItem(arxiv_id="2", title="P2", authors=[], url="u2", chunk_text="c2"),
            ],
            "retrieval_attempts": 1,
            "grading_results": [
                GradingResult(document_id="1", is_relevant=True, score=1.0),
                GradingResult(document_id="2", is_relevant=False, score=0.0),
            ],
            "relevant_sources": [
                SourceItem(arxiv_id="1", title="P1", authors=[], url="u1", chunk_text="c1"),
            ],
            "metadata": {},
        }

        service = AgenticRAGService(llm_provider=MagicMock())
        steps = service._extract_reasoning_steps(state)

        assert isinstance(steps, list)
        assert len(steps) > 0
        # Should mention guardrail, retrieval, grading
        combined = " ".join(steps).lower()
        assert "guardrail" in combined
        assert "retriev" in combined
        assert "grad" in combined

    def test_extract_reasoning_steps_empty_state(self):
        from unittest.mock import MagicMock

        from src.services.agents.agentic_rag import AgenticRAGService

        service = AgenticRAGService(llm_provider=MagicMock())
        state = {}
        steps = service._extract_reasoning_steps(state)
        assert isinstance(steps, list)


# ---------------------------------------------------------------------------
# Test: Factory (FR-5)
# ---------------------------------------------------------------------------


class TestFactory:
    """FR-5: create_agentic_rag_service factory."""

    def test_factory_creates_service(self, mock_llm_provider, mock_retrieval_pipeline):
        from src.services.agents.factory import create_agentic_rag_service

        service = create_agentic_rag_service(
            llm_provider=mock_llm_provider,
            retrieval_pipeline=mock_retrieval_pipeline,
        )

        from src.services.agents.agentic_rag import AgenticRAGService

        assert isinstance(service, AgenticRAGService)

    def test_factory_requires_llm_provider(self, mock_retrieval_pipeline):
        from src.services.agents.factory import create_agentic_rag_service

        with pytest.raises((ValueError, TypeError)):
            create_agentic_rag_service(
                llm_provider=None,
                retrieval_pipeline=mock_retrieval_pipeline,
            )

    def test_factory_optional_cache(self, mock_llm_provider, mock_retrieval_pipeline):
        from src.services.agents.factory import create_agentic_rag_service

        service = create_agentic_rag_service(
            llm_provider=mock_llm_provider,
            retrieval_pipeline=mock_retrieval_pipeline,
            cache_service=MagicMock(),
        )

        from src.services.agents.agentic_rag import AgenticRAGService

        assert isinstance(service, AgenticRAGService)


# ---------------------------------------------------------------------------
# Test: Out-of-scope node (still works via graph)
# ---------------------------------------------------------------------------


class TestOutOfScopeNode:
    """The out_of_scope node returns a polite rejection AIMessage."""

    def test_out_of_scope_node_function(self):
        from src.services.agents.agentic_rag import ainvoke_out_of_scope_step
        from src.services.agents.state import create_initial_state

        state = create_initial_state("How to make pizza?")
        result = ainvoke_out_of_scope_step(state)

        assert "messages" in result
        assert len(result["messages"]) == 1
        msg = result["messages"][0]
        assert isinstance(msg, AIMessage)
        assert "research assistant" in msg.content.lower()


# ---------------------------------------------------------------------------
# Test: AgenticRAGResponse model
# ---------------------------------------------------------------------------


class TestAgenticRAGResponse:
    """Response model validation."""

    def test_response_model_fields(self):
        from src.services.agents.agentic_rag import AgenticRAGResponse

        response = AgenticRAGResponse(
            answer="Test answer [1].",
            sources=[],
            reasoning_steps=["Step 1"],
            metadata={"key": "value"},
        )

        assert response.answer == "Test answer [1]."
        assert response.sources == []
        assert response.reasoning_steps == ["Step 1"]
        assert response.metadata == {"key": "value"}

    def test_response_model_defaults(self):
        from src.services.agents.agentic_rag import AgenticRAGResponse

        response = AgenticRAGResponse(answer="Answer")

        assert response.sources == []
        assert response.reasoning_steps == []
        assert response.metadata == {}
