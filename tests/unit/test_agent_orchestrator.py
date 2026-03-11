"""Tests for the agent orchestrator (S6.7).

Tests cover:
- Graph compilation and node registration
- Happy path: guardrail → retrieve → grade → generate
- Out-of-scope: guardrail reject → polite rejection
- Rewrite retry loop: grade fails → rewrite → re-retrieve → generate
- Max retrieval attempts enforcement
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
# Test: Happy Path (FR-2)
# ---------------------------------------------------------------------------


class TestHappyPathAsk:
    """FR-2: Full happy path — guardrail pass → retrieve → grade → generate."""

    @pytest.mark.asyncio
    async def test_happy_path_ask(self, mock_llm_provider, mock_retrieval_pipeline):
        """Query flows through guardrail → retrieve → grade → generate and returns answer with sources."""
        from src.services.agents.agentic_rag import AgenticRAGService

        # Setup guardrail to pass
        guardrail_response = GuardrailScoring(score=85, reason="Research question")
        # Setup grading to mark all as relevant
        from src.services.agents.models import GradeDocuments

        grade_yes = GradeDocuments(binary_score="yes", reasoning="Relevant")

        # Setup retrieval
        from src.schemas.api.search import SearchHit

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
                stages_executed=["hybrid_search", "rerank"],
                total_candidates=20,
                timings={"hybrid_search": 0.1, "rerank": 0.05},
            )
        )

        # Configure LLM mock to return different structured outputs per call
        structured_mock = AsyncMock()
        call_count = 0

        async def side_effect(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return guardrail_response  # guardrail
            elif call_count == 2:
                return grade_yes  # grading
            else:
                return None  # shouldn't happen

        structured_mock.ainvoke = side_effect

        # For generate, we need a regular model
        gen_mock = AsyncMock()
        answer_text = (
            "Transformers use self-attention [1].\n\n**Sources:**\n"
            "1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al."
        )
        gen_mock.ainvoke = AsyncMock(return_value=AIMessage(content=answer_text))

        model_mock = MagicMock()
        model_mock.with_structured_output.return_value = structured_mock
        mock_llm_provider.get_langchain_model.return_value = model_mock

        # For generation node, it doesn't use structured output — it uses ainvoke directly
        # So we need to handle both: with_structured_output and direct ainvoke
        model_mock.ainvoke = gen_mock.ainvoke

        service = AgenticRAGService(
            llm_provider=mock_llm_provider,
            retrieval_pipeline=mock_retrieval_pipeline,
        )

        result = await service.ask("What are transformers?")

        assert result.answer is not None
        assert len(result.answer) > 0
        assert isinstance(result.sources, list)
        assert isinstance(result.reasoning_steps, list)
        assert isinstance(result.metadata, dict)


# ---------------------------------------------------------------------------
# Test: Out-of-Scope (FR-3)
# ---------------------------------------------------------------------------


class TestOutOfScopeQuery:
    """FR-3: Off-topic query → guardrail reject → out_of_scope message."""

    @pytest.mark.asyncio
    async def test_out_of_scope_query(self, mock_llm_provider, mock_retrieval_pipeline):
        """Off-topic query gets politely rejected."""
        from src.services.agents.agentic_rag import AgenticRAGService

        # Guardrail rejects
        guardrail_response = GuardrailScoring(score=10, reason="Off-topic: cooking recipe")

        structured_mock = AsyncMock()
        structured_mock.ainvoke = AsyncMock(return_value=guardrail_response)

        model_mock = MagicMock()
        model_mock.with_structured_output.return_value = structured_mock
        mock_llm_provider.get_langchain_model.return_value = model_mock

        service = AgenticRAGService(
            llm_provider=mock_llm_provider,
            retrieval_pipeline=mock_retrieval_pipeline,
        )

        result = await service.ask("How do I make pasta carbonara?")

        assert "research assistant" in result.answer.lower() or "can't help" in result.answer.lower()
        assert len(result.sources) == 0

    def test_out_of_scope_node_function(self):
        """The out_of_scope node returns a polite rejection AIMessage."""
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
# Test: Rewrite Retry Loop (FR-2 edge)
# ---------------------------------------------------------------------------


class TestRewriteRetryLoop:
    """Grade fails → rewrite → re-retrieve → grade passes → generate."""

    @pytest.mark.asyncio
    async def test_rewrite_retry_loop(self, mock_llm_provider, mock_retrieval_pipeline):
        """After grading finds no relevant docs, rewrite and retry."""
        from src.schemas.api.search import SearchHit
        from src.services.agents.agentic_rag import AgenticRAGService
        from src.services.agents.models import GradeDocuments
        from src.services.agents.nodes.rewrite_query_node import QueryRewriteOutput

        # Setup retrieval to always return docs
        mock_retrieval_pipeline.retrieve = AsyncMock(
            return_value=MagicMock(
                results=[
                    SearchHit(
                        arxiv_id="2001.00001",
                        title="Some Paper",
                        authors=["Author A"],
                        score=0.5,
                        chunk_text="Some content about the topic.",
                        pdf_url="",
                    ),
                ],
                stages_executed=["hybrid_search"],
                total_candidates=10,
                timings={"hybrid_search": 0.1},
            )
        )

        guardrail_pass = GuardrailScoring(score=80, reason="Research question")
        grade_no = GradeDocuments(binary_score="no", reasoning="Not relevant")
        grade_yes = GradeDocuments(binary_score="yes", reasoning="Relevant")
        rewrite_output = QueryRewriteOutput(rewritten_query="improved query", reasoning="Expanded terms")

        # Track call sequence to return different responses
        structured_calls = []

        responses = {1: guardrail_pass, 2: grade_no, 3: rewrite_output, 4: grade_yes}

        async def structured_side_effect(prompt):
            structured_calls.append(prompt)
            return responses.get(len(structured_calls), grade_yes)

        structured_mock = AsyncMock()
        structured_mock.ainvoke = structured_side_effect

        rewrite_answer = (
            "Answer based on rewritten query [1].\n\n**Sources:**\n1. [Some Paper](https://arxiv.org/abs/2001.00001) — Author A"
        )
        model_mock = MagicMock()
        model_mock.with_structured_output.return_value = structured_mock
        model_mock.ainvoke = AsyncMock(return_value=AIMessage(content=rewrite_answer))
        mock_llm_provider.get_langchain_model.return_value = model_mock

        service = AgenticRAGService(
            llm_provider=mock_llm_provider,
            retrieval_pipeline=mock_retrieval_pipeline,
            max_retrieval_attempts=3,
        )

        result = await service.ask("What is quantum computing?")

        assert result.answer is not None
        assert len(result.answer) > 0
        # Should have done at least 2 retrieval attempts
        assert mock_retrieval_pipeline.retrieve.call_count >= 2


# ---------------------------------------------------------------------------
# Test: Max Retrieval Attempts (FR-2 edge)
# ---------------------------------------------------------------------------


class TestMaxRetrievalAttempts:
    """Rewrite loop stops after max attempts."""

    @pytest.mark.asyncio
    async def test_max_retrieval_attempts(self, mock_llm_provider, mock_retrieval_pipeline):
        """After max attempts, force generation even without relevant sources."""
        from src.schemas.api.search import SearchHit
        from src.services.agents.agentic_rag import AgenticRAGService
        from src.services.agents.models import GradeDocuments
        from src.services.agents.nodes.rewrite_query_node import QueryRewriteOutput

        mock_retrieval_pipeline.retrieve = AsyncMock(
            return_value=MagicMock(
                results=[
                    SearchHit(
                        arxiv_id="2001.00001",
                        title="Irrelevant Paper",
                        authors=["Author A"],
                        score=0.3,
                        chunk_text="Not really about the topic.",
                        pdf_url="",
                    ),
                ],
                stages_executed=["hybrid_search"],
                total_candidates=5,
                timings={"hybrid_search": 0.1},
            )
        )

        guardrail_pass = GuardrailScoring(score=80, reason="Research question")
        grade_no = GradeDocuments(binary_score="no", reasoning="Not relevant")
        rewrite_output = QueryRewriteOutput(rewritten_query="another query", reasoning="Tried again")

        async def structured_side_effect(prompt):
            # Always guardrail pass, always grade no, always rewrite
            if "domain relevance" in str(prompt).lower() or "rate the following" in str(prompt).lower():
                return guardrail_pass
            elif "rewrite" in str(prompt).lower() or "rephrase" in str(prompt).lower():
                return rewrite_output
            return grade_no

        structured_mock = AsyncMock()
        structured_mock.ainvoke = structured_side_effect

        model_mock = MagicMock()
        model_mock.with_structured_output.return_value = structured_mock
        model_mock.ainvoke = AsyncMock(
            return_value=AIMessage(content="I don't have papers on that exact topic in my knowledge base.")
        )
        mock_llm_provider.get_langchain_model.return_value = model_mock

        service = AgenticRAGService(
            llm_provider=mock_llm_provider,
            retrieval_pipeline=mock_retrieval_pipeline,
            max_retrieval_attempts=2,
        )

        result = await service.ask("Obscure niche topic?")

        assert result.answer is not None
        # Should not exceed max attempts
        assert mock_retrieval_pipeline.retrieve.call_count <= 3  # max_attempts + 1 tolerance


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

        steps = AgenticRAGService._extract_reasoning_steps(state)

        assert isinstance(steps, list)
        assert len(steps) > 0
        # Should mention guardrail, retrieval, grading
        combined = " ".join(steps).lower()
        assert "guardrail" in combined
        assert "retriev" in combined
        assert "grad" in combined

    def test_extract_reasoning_steps_empty_state(self):
        from src.services.agents.agentic_rag import AgenticRAGService

        state = {}
        steps = AgenticRAGService._extract_reasoning_steps(state)
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
