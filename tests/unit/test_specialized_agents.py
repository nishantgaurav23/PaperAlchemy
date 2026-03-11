"""Tests for specialized agents (S6.8).

Tests cover:
- Base protocol contract (SpecializedAgentBase)
- SummarizerAgent: structured summaries with all 5 sections
- FactCheckerAgent: claim verification with evidence
- TrendAnalyzerAgent: trend identification with direction
- CitationTrackerAgent: citation relationship mapping
- AgentRegistry: dispatch and fallback behavior
- All agents cite sources with arXiv links
- All agents use retrieval pipeline
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from src.services.agents.context import AgentContext
from src.services.agents.models import SourceItem
from src.services.agents.specialized import AgentRegistry
from src.services.agents.specialized.base import SpecializedAgentBase, SpecializedAgentResult
from src.services.agents.specialized.citation_tracker import CitationTrackerAgent, CitationTrackResult
from src.services.agents.specialized.fact_checker import ClaimVerification, FactCheckerAgent, FactCheckResult
from src.services.agents.specialized.summarizer import SummarizerAgent, SummarizerResult
from src.services.agents.specialized.trend_analyzer import TrendAnalysisResult, TrendAnalyzerAgent, TrendItem

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_llm_for_structured_output(context: AgentContext, return_value):
    """Configure context.llm_provider to return a specific structured output from ainvoke."""
    mock_model = MagicMock()
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(return_value=return_value)
    mock_model.with_structured_output.return_value = mock_structured
    context.llm_provider.get_langchain_model.return_value = mock_model


def _mock_llm_for_structured_error(context: AgentContext, error: Exception):
    """Configure context.llm_provider to raise an error from ainvoke."""
    mock_model = MagicMock()
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(side_effect=error)
    mock_model.with_structured_output.return_value = mock_structured
    context.llm_provider.get_langchain_model.return_value = mock_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider with get_langchain_model."""
    provider = MagicMock()
    mock_model = MagicMock()
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock()
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
        top_k=5,
    )


@pytest.fixture
def sample_sources() -> list[SourceItem]:
    """Create realistic sample papers for testing."""
    return [
        SourceItem(
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            authors=["Vaswani", "Shazeer", "Parmar"],
            url="https://arxiv.org/abs/1706.03762",
            relevance_score=0.95,
            chunk_text="We propose a new architecture based entirely on attention mechanisms...",
        ),
        SourceItem(
            arxiv_id="1810.04805",
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            authors=["Devlin", "Chang", "Lee", "Toutanova"],
            url="https://arxiv.org/abs/1810.04805",
            relevance_score=0.88,
            chunk_text="We introduce BERT, a new language representation model...",
        ),
        SourceItem(
            arxiv_id="2005.14165",
            title="Language Models are Few-Shot Learners",
            authors=["Brown", "Mann", "Ryder"],
            url="https://arxiv.org/abs/2005.14165",
            relevance_score=0.82,
            chunk_text="Recent work demonstrates that large language models can achieve strong performance...",
        ),
    ]


# ---------------------------------------------------------------------------
# Test: Base Protocol (FR-1)
# ---------------------------------------------------------------------------


class TestSpecializedAgentBase:
    """Tests for the base protocol contract."""

    def test_base_is_abstract(self):
        """SpecializedAgentBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            SpecializedAgentBase()

    def test_specialized_agent_result_model(self):
        """SpecializedAgentResult has correct fields."""
        result = SpecializedAgentResult(
            agent_name="test",
            analysis="Some analysis",
            sources=[],
            metadata={"key": "value"},
        )
        assert result.agent_name == "test"
        assert result.analysis == "Some analysis"
        assert result.sources == []
        assert result.metadata == {"key": "value"}

    def test_specialized_agent_result_defaults(self):
        """SpecializedAgentResult has sensible defaults."""
        result = SpecializedAgentResult(agent_name="test", analysis="test")
        assert result.sources == []
        assert result.metadata == {}


# ---------------------------------------------------------------------------
# Test: SummarizerAgent (FR-2)
# ---------------------------------------------------------------------------


class TestSummarizerAgent:
    """Tests for the paper summarizer agent."""

    @pytest.mark.asyncio
    async def test_summarizer_produces_structured_output(self, context, sample_sources):
        """Summarizer returns all 5 sections."""
        agent = SummarizerAgent()

        _mock_llm_for_structured_output(
            context,
            SummarizerResult(
                objective="Study self-attention mechanisms for sequence transduction.",
                methodology="Replace recurrence with multi-head attention in encoder-decoder.",
                key_findings=["Outperforms RNNs on translation", "Trains faster"],
                contributions=["Transformer architecture", "Multi-head attention"],
                limitations=["Quadratic memory in sequence length"],
                summary_text="The paper proposes the Transformer architecture...",
                sources=sample_sources,
            ),
        )

        result = await agent.run(
            query="Summarize the Attention Is All You Need paper",
            context=context,
            papers=sample_sources,
        )

        assert isinstance(result, SummarizerResult)
        assert result.objective != ""
        assert result.methodology != ""
        assert len(result.key_findings) > 0
        assert len(result.contributions) > 0
        assert len(result.limitations) > 0
        assert result.summary_text != ""

    @pytest.mark.asyncio
    async def test_summarizer_handles_no_papers(self, context):
        """Summarizer retrieves its own papers when none provided."""
        agent = SummarizerAgent()

        # Mock retrieval to return some results
        mock_retrieve_result = MagicMock()
        mock_retrieve_result.results = [
            MagicMock(
                arxiv_id="1706.03762",
                title="Attention Is All You Need",
                authors=["Vaswani"],
                pdf_url="https://arxiv.org/abs/1706.03762",
                score=0.9,
                chunk_text="We propose a new architecture...",
            )
        ]
        context.retrieval_pipeline.retrieve = AsyncMock(return_value=mock_retrieve_result)

        _mock_llm_for_structured_output(
            context,
            SummarizerResult(
                objective="Study attention mechanisms.",
                methodology="Replace recurrence.",
                key_findings=["Better results"],
                contributions=["Transformer"],
                limitations=["Memory"],
                summary_text="The paper proposes...",
                sources=[
                    SourceItem(
                        arxiv_id="1706.03762",
                        title="Attention Is All You Need",
                        authors=["Vaswani"],
                        url="https://arxiv.org/abs/1706.03762",
                        relevance_score=0.9,
                        chunk_text="We propose a new architecture...",
                    )
                ],
            ),
        )

        result = await agent.run(
            query="Summarize attention mechanisms research",
            context=context,
        )

        assert isinstance(result, SummarizerResult)
        context.retrieval_pipeline.retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarizer_cites_sources(self, context, sample_sources):
        """Summarizer includes sources with arXiv links."""
        agent = SummarizerAgent()

        _mock_llm_for_structured_output(
            context,
            SummarizerResult(
                objective="Test",
                methodology="Test",
                key_findings=["Test"],
                contributions=["Test"],
                limitations=["Test"],
                summary_text="Test summary",
                sources=sample_sources,
            ),
        )

        result = await agent.run(query="Test", context=context, papers=sample_sources)

        assert len(result.sources) > 0
        for source in result.sources:
            assert "arxiv.org" in source.url

    @pytest.mark.asyncio
    async def test_summarizer_llm_failure_graceful(self, context, sample_sources):
        """Summarizer handles LLM failure gracefully."""
        agent = SummarizerAgent()

        _mock_llm_for_structured_error(context, Exception("LLM timeout"))

        result = await agent.run(query="Test", context=context, papers=sample_sources)

        assert isinstance(result, SummarizerResult)
        assert "error" in result.summary_text.lower() or result.objective == ""

    @pytest.mark.asyncio
    async def test_summarizer_empty_papers_returns_message(self, context):
        """Summarizer returns a message when no papers are found."""
        agent = SummarizerAgent()

        # Mock retrieval returning empty
        mock_retrieve_result = MagicMock()
        mock_retrieve_result.results = []
        context.retrieval_pipeline.retrieve = AsyncMock(return_value=mock_retrieve_result)

        result = await agent.run(query="Summarize quantum computing", context=context)

        assert isinstance(result, SummarizerResult)
        assert "no relevant papers" in result.summary_text.lower()


# ---------------------------------------------------------------------------
# Test: FactCheckerAgent (FR-3)
# ---------------------------------------------------------------------------


class TestFactCheckerAgent:
    """Tests for the fact-checker agent."""

    @pytest.mark.asyncio
    async def test_fact_checker_supported_claim(self, context, sample_sources):
        """Fact-checker returns 'supported' with evidence."""
        agent = FactCheckerAgent()

        _mock_llm_for_structured_output(
            context,
            FactCheckResult(
                claims=[
                    ClaimVerification(
                        claim="Transformers outperform RNNs on machine translation",
                        verdict="supported",
                        evidence=[sample_sources[0]],
                        explanation="Vaswani et al. (2017) show BLEU score improvements...",
                    )
                ],
                sources=sample_sources[:1],
                overall_assessment="The claim is well-supported by the literature.",
            ),
        )

        result = await agent.run(
            query="Verify: Transformers outperform RNNs on machine translation",
            context=context,
            papers=sample_sources,
        )

        assert isinstance(result, FactCheckResult)
        assert len(result.claims) > 0
        assert result.claims[0].verdict == "supported"
        assert len(result.claims[0].evidence) > 0

    @pytest.mark.asyncio
    async def test_fact_checker_contradicted_claim(self, context, sample_sources):
        """Fact-checker returns 'contradicted' with evidence."""
        agent = FactCheckerAgent()

        _mock_llm_for_structured_output(
            context,
            FactCheckResult(
                claims=[
                    ClaimVerification(
                        claim="RNNs are always better than Transformers",
                        verdict="contradicted",
                        evidence=[sample_sources[0]],
                        explanation="Evidence shows Transformers outperform RNNs.",
                    )
                ],
                sources=sample_sources[:1],
                overall_assessment="The claim is contradicted.",
            ),
        )

        result = await agent.run(
            query="Verify: RNNs are always better than Transformers",
            context=context,
            papers=sample_sources,
        )

        assert result.claims[0].verdict == "contradicted"

    @pytest.mark.asyncio
    async def test_fact_checker_insufficient_evidence(self, context, sample_sources):
        """Fact-checker returns 'insufficient_evidence' when no evidence found."""
        agent = FactCheckerAgent()

        _mock_llm_for_structured_output(
            context,
            FactCheckResult(
                claims=[
                    ClaimVerification(
                        claim="Quantum computing will replace classical computing by 2030",
                        verdict="insufficient_evidence",
                        evidence=[],
                        explanation="No papers in the knowledge base address this specific claim.",
                    )
                ],
                sources=[],
                overall_assessment="Insufficient evidence to verify.",
            ),
        )

        result = await agent.run(
            query="Verify: Quantum computing will replace classical computing by 2030",
            context=context,
            papers=sample_sources,
        )

        assert result.claims[0].verdict == "insufficient_evidence"

    @pytest.mark.asyncio
    async def test_fact_checker_uses_retrieval(self, context):
        """Fact-checker calls retrieval when no papers provided."""
        agent = FactCheckerAgent()

        mock_retrieve_result = MagicMock()
        mock_retrieve_result.results = []
        context.retrieval_pipeline.retrieve = AsyncMock(return_value=mock_retrieve_result)

        await agent.run(query="Verify some claim", context=context)

        context.retrieval_pipeline.retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_fact_checker_llm_failure_graceful(self, context, sample_sources):
        """Fact-checker handles LLM failure gracefully."""
        agent = FactCheckerAgent()

        _mock_llm_for_structured_error(context, Exception("LLM error"))

        result = await agent.run(query="Verify claim", context=context, papers=sample_sources)

        assert isinstance(result, FactCheckResult)
        assert len(result.claims) > 0
        assert result.claims[0].verdict == "insufficient_evidence"


# ---------------------------------------------------------------------------
# Test: TrendAnalyzerAgent (FR-4)
# ---------------------------------------------------------------------------


class TestTrendAnalyzerAgent:
    """Tests for the trend analyzer agent."""

    @pytest.mark.asyncio
    async def test_trend_analyzer_identifies_trends(self, context, sample_sources):
        """Trend analyzer returns trends with direction."""
        agent = TrendAnalyzerAgent()

        _mock_llm_for_structured_output(
            context,
            TrendAnalysisResult(
                trends=[
                    TrendItem(
                        topic="Transformer architectures",
                        direction="rising",
                        key_papers=[sample_sources[0]],
                        description="Self-attention mechanisms are increasingly dominant.",
                    ),
                    TrendItem(
                        topic="Pre-training large language models",
                        direction="rising",
                        key_papers=[sample_sources[1], sample_sources[2]],
                        description="Pre-training then fine-tuning is the standard paradigm.",
                    ),
                ],
                timeline="2017: Transformer introduced. 2018: BERT. 2020: GPT-3.",
                emerging_topics=["Efficient transformers", "Multimodal models"],
                sources=sample_sources,
            ),
        )

        result = await agent.run(
            query="What are the trends in NLP research?",
            context=context,
            papers=sample_sources,
        )

        assert isinstance(result, TrendAnalysisResult)
        assert len(result.trends) > 0
        assert result.trends[0].direction in ("rising", "stable", "declining")
        assert len(result.trends[0].key_papers) > 0

    @pytest.mark.asyncio
    async def test_trend_analyzer_emerging_topics(self, context, sample_sources):
        """Trend analyzer identifies emerging topics."""
        agent = TrendAnalyzerAgent()

        _mock_llm_for_structured_output(
            context,
            TrendAnalysisResult(
                trends=[],
                timeline="",
                emerging_topics=["Vision transformers", "Instruction tuning"],
                sources=sample_sources,
            ),
        )

        result = await agent.run(
            query="What are emerging topics?",
            context=context,
            papers=sample_sources,
        )

        assert len(result.emerging_topics) > 0

    @pytest.mark.asyncio
    async def test_trend_analyzer_uses_retrieval(self, context):
        """Trend analyzer calls retrieval when no papers provided."""
        agent = TrendAnalyzerAgent()

        mock_retrieve_result = MagicMock()
        mock_retrieve_result.results = []
        context.retrieval_pipeline.retrieve = AsyncMock(return_value=mock_retrieve_result)

        await agent.run(query="Trends in NLP", context=context)

        context.retrieval_pipeline.retrieve.assert_called_once()


# ---------------------------------------------------------------------------
# Test: CitationTrackerAgent (FR-5)
# ---------------------------------------------------------------------------


class TestCitationTrackerAgent:
    """Tests for the citation tracker agent."""

    @pytest.mark.asyncio
    async def test_citation_tracker_finds_citations(self, context, sample_sources):
        """Citation tracker maps citation relationships."""
        agent = CitationTrackerAgent()

        _mock_llm_for_structured_output(
            context,
            CitationTrackResult(
                target_paper=sample_sources[0],
                cited_by=[sample_sources[1], sample_sources[2]],
                references=[],
                citation_count=2,
                influence_summary="Highly influential — cited by BERT and GPT-3.",
                sources=sample_sources,
            ),
        )

        result = await agent.run(
            query="What papers cite Attention Is All You Need?",
            context=context,
            papers=sample_sources,
        )

        assert isinstance(result, CitationTrackResult)
        assert result.target_paper.arxiv_id == "1706.03762"
        assert len(result.cited_by) > 0
        assert result.citation_count >= len(result.cited_by)

    @pytest.mark.asyncio
    async def test_citation_tracker_paper_not_found(self, context):
        """Citation tracker handles paper not in knowledge base."""
        agent = CitationTrackerAgent()

        mock_retrieve_result = MagicMock()
        mock_retrieve_result.results = []
        context.retrieval_pipeline.retrieve = AsyncMock(return_value=mock_retrieve_result)

        result = await agent.run(
            query="What papers cite some unknown paper?",
            context=context,
        )

        assert result.citation_count == 0
        assert len(result.cited_by) == 0

    @pytest.mark.asyncio
    async def test_citation_tracker_cites_sources(self, context, sample_sources):
        """Citation tracker includes arXiv links in output."""
        agent = CitationTrackerAgent()

        _mock_llm_for_structured_output(
            context,
            CitationTrackResult(
                target_paper=sample_sources[0],
                cited_by=[sample_sources[1]],
                references=[],
                citation_count=1,
                influence_summary="Cited by BERT.",
                sources=sample_sources[:2],
            ),
        )

        result = await agent.run(query="Citations for Attention", context=context, papers=sample_sources)

        for source in result.sources:
            assert "arxiv.org" in source.url


# ---------------------------------------------------------------------------
# Test: AgentRegistry (FR-6)
# ---------------------------------------------------------------------------


class TestAgentRegistry:
    """Tests for agent registry and dispatch."""

    def test_registry_has_all_agents(self):
        """Registry contains all 4 specialized agents."""
        registry = AgentRegistry()
        assert "summarizer" in registry.agent_names
        assert "fact_checker" in registry.agent_names
        assert "trend_analyzer" in registry.agent_names
        assert "citation_tracker" in registry.agent_names

    def test_registry_dispatch_summarizer(self):
        """Registry dispatches to SummarizerAgent."""
        registry = AgentRegistry()
        agent = registry.get("summarizer")
        assert isinstance(agent, SummarizerAgent)

    def test_registry_dispatch_fact_checker(self):
        """Registry dispatches to FactCheckerAgent."""
        registry = AgentRegistry()
        agent = registry.get("fact_checker")
        assert isinstance(agent, FactCheckerAgent)

    def test_registry_dispatch_trend_analyzer(self):
        """Registry dispatches to TrendAnalyzerAgent."""
        registry = AgentRegistry()
        agent = registry.get("trend_analyzer")
        assert isinstance(agent, TrendAnalyzerAgent)

    def test_registry_dispatch_citation_tracker(self):
        """Registry dispatches to CitationTrackerAgent."""
        registry = AgentRegistry()
        agent = registry.get("citation_tracker")
        assert isinstance(agent, CitationTrackerAgent)

    def test_registry_unknown_type_returns_none(self):
        """Registry returns None for unknown agent type."""
        registry = AgentRegistry()
        assert registry.get("unknown_agent") is None

    def test_registry_agent_names_list(self):
        """Registry exposes list of available agent names."""
        registry = AgentRegistry()
        names = registry.agent_names
        assert len(names) == 4
        assert isinstance(names, list)


# ---------------------------------------------------------------------------
# Test: Cross-cutting concerns
# ---------------------------------------------------------------------------


class TestAllAgentsCrossCutting:
    """Tests that apply to all specialized agents."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("agent_class", [SummarizerAgent, FactCheckerAgent, TrendAnalyzerAgent, CitationTrackerAgent])
    async def test_all_agents_have_run_method(self, agent_class):
        """All agents implement the run() method."""
        agent = agent_class()
        assert hasattr(agent, "run")
        assert callable(agent.run)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("agent_class", [SummarizerAgent, FactCheckerAgent, TrendAnalyzerAgent, CitationTrackerAgent])
    async def test_all_agents_have_name(self, agent_class):
        """All agents have a name property."""
        agent = agent_class()
        assert hasattr(agent, "name")
        assert isinstance(agent.name, str)
        assert len(agent.name) > 0
