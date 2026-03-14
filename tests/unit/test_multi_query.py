"""Unit tests for Multi-Query retrieval service.

Tests cover:
- Query variation generation via LLM
- Parallel hybrid search across variations
- Result deduplication by chunk_id
- RRF fusion scoring across query results
- Full multi-query retrieval flow
- Graceful fallback on LLM/search failures
- Edge cases (empty query, empty results, parsing)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from src.services.llm.provider import LLMResponse, UsageMetadata


def _make_hit(arxiv_id: str, chunk_id: str, title: str, score: float, **kwargs) -> dict:
    """Helper to create a search hit dict."""
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": kwargs.get("authors", ["Author A"]),
        "abstract": kwargs.get("abstract", "Abstract text"),
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        "chunk_text": kwargs.get("chunk_text", f"Chunk text for {chunk_id}"),
        "chunk_id": chunk_id,
        "section_title": kwargs.get("section_title", "Introduction"),
        "score": score,
        "highlights": {},
    }


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider that returns numbered query variations."""
    provider = AsyncMock()
    provider.generate = AsyncMock(
        return_value=LLMResponse(
            text=(
                "1. How do transformer neural networks work in natural language processing?\n"
                "2. What is the architecture of transformer models for NLP tasks?\n"
                "3. Self-attention mechanisms in transformer-based language models"
            ),
            model="gemini-2.0-flash",
            provider="gemini",
            usage=UsageMetadata(prompt_tokens=50, completion_tokens=60, total_tokens=110),
        )
    )
    return provider


@pytest.fixture
def mock_embeddings_client():
    """Mock Jina embeddings client that returns fake vectors."""
    client = AsyncMock()
    client.embed_query = AsyncMock(return_value=[0.1] * 1024)
    return client


@pytest.fixture
def mock_opensearch_client():
    """Mock OpenSearch client with hybrid search (sync + async methods)."""
    client = MagicMock()

    _default_return = {
        "total": 2,
        "hits": [
            _make_hit("1706.03762", "1706.03762_chunk_0", "Attention Is All You Need", 0.95),
            _make_hit("1810.04805", "1810.04805_chunk_0", "BERT", 0.88),
        ],
    }

    # Sync mock kept for backward compat
    client.search_chunks_hybrid = MagicMock(return_value=_default_return)
    # Async mock used by production code
    client.asearch_chunks_hybrid = AsyncMock(return_value=_default_return)
    return client


@pytest.fixture
def multi_query_settings():
    """MultiQuery settings for testing."""
    from src.config import MultiQuerySettings

    return MultiQuerySettings(enabled=True, num_queries=3, temperature=0.7, max_tokens=300, rrf_k=60)


@pytest.fixture
def multi_query_service(multi_query_settings, mock_llm_provider, mock_embeddings_client, mock_opensearch_client):
    """Fully constructed MultiQueryService with mocked dependencies."""
    from src.services.retrieval.multi_query import MultiQueryService

    return MultiQueryService(
        settings=multi_query_settings,
        llm_provider=mock_llm_provider,
        embeddings_client=mock_embeddings_client,
        opensearch_client=mock_opensearch_client,
    )


class TestGenerateQueryVariations:
    """Tests for FR-2: Query variation generation."""

    @pytest.mark.asyncio
    async def test_generate_query_variations(self, multi_query_service, mock_llm_provider):
        """Verify LLM called with correct prompt, returns list of variations."""
        result = await multi_query_service.generate_query_variations("What are transformers in NLP?")

        assert isinstance(result, list)
        assert len(result) == 3
        mock_llm_provider.generate.assert_called_once()

        # Prompt should contain the original query
        prompt = mock_llm_provider.generate.call_args[0][0]
        assert "What are transformers in NLP?" in prompt

    @pytest.mark.asyncio
    async def test_generate_query_variations_empty_query(self, multi_query_service):
        """Verify ValueError raised for empty query."""
        with pytest.raises(ValueError, match="[Qq]uery.*empty|[Ee]mpty.*query"):
            await multi_query_service.generate_query_variations("")

    @pytest.mark.asyncio
    async def test_generate_query_variations_whitespace_query(self, multi_query_service):
        """Verify ValueError raised for whitespace-only query."""
        with pytest.raises(ValueError, match="[Qq]uery.*empty|[Ee]mpty.*query"):
            await multi_query_service.generate_query_variations("   ")

    @pytest.mark.asyncio
    async def test_generate_query_variations_llm_failure(self, multi_query_service, mock_llm_provider):
        """Verify fallback returns [original_query] when LLM fails."""
        mock_llm_provider.generate.side_effect = Exception("LLM service unavailable")

        result = await multi_query_service.generate_query_variations("What are transformers?")

        assert result == ["What are transformers?"]

    @pytest.mark.asyncio
    async def test_generate_query_variations_empty_response(self, multi_query_service, mock_llm_provider):
        """Verify fallback returns [original_query] when LLM returns empty."""
        mock_llm_provider.generate.return_value = LLMResponse(text="", model="gemini-2.0-flash", provider="gemini")

        result = await multi_query_service.generate_query_variations("What are transformers?")

        assert result == ["What are transformers?"]

    @pytest.mark.asyncio
    async def test_generate_query_variations_parsing_numbered(self, multi_query_service, mock_llm_provider):
        """Verify numbered list parsing handles various formats."""
        mock_llm_provider.generate.return_value = LLMResponse(
            text="1. First variation\n2) Second variation\n3 - Third variation",
            model="gemini-2.0-flash",
            provider="gemini",
        )

        result = await multi_query_service.generate_query_variations("test query")

        assert len(result) >= 2  # At least some should parse

    @pytest.mark.asyncio
    async def test_generate_query_variations_parsing_bullets(self, multi_query_service, mock_llm_provider):
        """Verify bullet-point list parsing."""
        mock_llm_provider.generate.return_value = LLMResponse(
            text="- First variation\n- Second variation\n- Third variation",
            model="gemini-2.0-flash",
            provider="gemini",
        )

        result = await multi_query_service.generate_query_variations("test query")

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_generate_query_variations_uses_temperature(self, multi_query_service, mock_llm_provider):
        """Verify temperature=0.7 is used for creative variations."""
        await multi_query_service.generate_query_variations("test query")

        call_kwargs = mock_llm_provider.generate.call_args[1]
        assert call_kwargs.get("temperature") == 0.7

    @pytest.mark.asyncio
    async def test_generate_query_variations_prompt_content(self, multi_query_service, mock_llm_provider):
        """Verify the prompt asks for diverse variations."""
        await multi_query_service.generate_query_variations("What is attention mechanism?")

        prompt = mock_llm_provider.generate.call_args[0][0]
        prompt_lower = prompt.lower()
        assert any(word in prompt_lower for word in ["variation", "reformulation", "rephrase", "alternative", "different"])


class TestRetrieveWithMultiQuery:
    """Tests for FR-1, FR-3, FR-4, FR-5: Full multi-query retrieval flow."""

    @pytest.mark.asyncio
    async def test_retrieve_with_multi_query_full_flow(
        self, multi_query_service, mock_llm_provider, mock_embeddings_client, mock_opensearch_client
    ):
        """Verify end-to-end: generate variations → parallel search → deduplicate → fuse → return."""
        from src.services.retrieval.multi_query import MultiQueryResult

        result = await multi_query_service.retrieve_with_multi_query("What are transformers in NLP?")

        assert isinstance(result, MultiQueryResult)
        assert result.original_query == "What are transformers in NLP?"
        assert len(result.generated_queries) == 3
        assert len(result.results) > 0

        # LLM called once for variations
        mock_llm_provider.generate.assert_called_once()
        # Embeddings called for each variation
        assert mock_embeddings_client.embed_query.call_count == 3
        # OpenSearch called for each variation
        assert mock_opensearch_client.asearch_chunks_hybrid.call_count == 3

    @pytest.mark.asyncio
    async def test_retrieve_with_multi_query_empty_query(self, multi_query_service):
        """Verify ValueError for empty query."""
        with pytest.raises(ValueError, match="[Qq]uery.*empty|[Ee]mpty.*query"):
            await multi_query_service.retrieve_with_multi_query("")

    @pytest.mark.asyncio
    async def test_retrieve_with_multi_query_deduplication(self, multi_query_service, mock_opensearch_client):
        """Verify same chunk_id across queries is deduplicated."""
        # All 3 query variations return the same chunk
        _dedup_return = {
            "total": 1,
            "hits": [_make_hit("1706.03762", "1706.03762_chunk_0", "Attention Is All You Need", 0.95)],
        }
        mock_opensearch_client.search_chunks_hybrid = MagicMock(return_value=_dedup_return)
        mock_opensearch_client.asearch_chunks_hybrid = AsyncMock(return_value=_dedup_return)

        result = await multi_query_service.retrieve_with_multi_query("transformers")

        # Should have exactly 1 result after dedup
        chunk_ids = [hit.chunk_id for hit in result.results]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk_ids found"

    @pytest.mark.asyncio
    async def test_retrieve_with_multi_query_rrf_fusion(
        self, multi_query_service, mock_opensearch_client, mock_embeddings_client
    ):
        """Verify RRF scoring ranks results correctly."""
        # Query 1: doc A rank 1, doc B rank 2
        # Query 2: doc B rank 1, doc A rank 2
        # Query 3: doc A rank 1, doc C rank 2
        # Expected: doc A has highest RRF score (appears rank 1 in 2/3 queries)
        call_count = 0

        def search_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "total": 2,
                    "hits": [
                        _make_hit("A", "A_chunk_0", "Paper A", 0.95),
                        _make_hit("B", "B_chunk_0", "Paper B", 0.85),
                    ],
                }
            elif call_count == 2:
                return {
                    "total": 2,
                    "hits": [
                        _make_hit("B", "B_chunk_0", "Paper B", 0.92),
                        _make_hit("A", "A_chunk_0", "Paper A", 0.80),
                    ],
                }
            else:
                return {
                    "total": 2,
                    "hits": [
                        _make_hit("A", "A_chunk_0", "Paper A", 0.90),
                        _make_hit("C", "C_chunk_0", "Paper C", 0.70),
                    ],
                }

        mock_opensearch_client.search_chunks_hybrid = MagicMock(side_effect=search_side_effect)
        mock_opensearch_client.asearch_chunks_hybrid = AsyncMock(side_effect=search_side_effect)

        result = await multi_query_service.retrieve_with_multi_query("test query")

        # Doc A should rank first (appears rank 1 in queries 1 and 3)
        assert result.results[0].chunk_id == "A_chunk_0"
        # Doc B should rank second
        assert result.results[1].chunk_id == "B_chunk_0"
        # Doc C last
        assert result.results[2].chunk_id == "C_chunk_0"

    @pytest.mark.asyncio
    async def test_retrieve_with_multi_query_fallback_on_generation_error(
        self, multi_query_service, mock_llm_provider, mock_embeddings_client, mock_opensearch_client
    ):
        """Verify falls back to single query search when LLM fails."""
        mock_llm_provider.generate.side_effect = Exception("LLM unavailable")

        result = await multi_query_service.retrieve_with_multi_query("What are transformers?")

        # Should still return results (fallback to original query)
        assert len(result.results) > 0
        assert result.generated_queries == ["What are transformers?"]
        # Only one search (for the original query)
        assert mock_embeddings_client.embed_query.call_count == 1
        assert mock_opensearch_client.asearch_chunks_hybrid.call_count == 1

    @pytest.mark.asyncio
    async def test_retrieve_with_multi_query_partial_search_failure(
        self, multi_query_service, mock_embeddings_client, mock_opensearch_client
    ):
        """Verify continues when some queries fail during search."""
        call_count = 0

        def embed_side_effect(text):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("Jina API error for query 2")
            return [0.1] * 1024

        mock_embeddings_client.embed_query = AsyncMock(side_effect=embed_side_effect)

        result = await multi_query_service.retrieve_with_multi_query("test query")

        # Should still return results from the successful queries
        assert len(result.results) > 0

    @pytest.mark.asyncio
    async def test_retrieve_with_multi_query_respects_top_k(self, multi_query_service, mock_opensearch_client):
        """Verify top_k parameter limits final results."""
        # Return many hits per query
        _topk_return = {
            "total": 5,
            "hits": [_make_hit(f"paper_{i}", f"paper_{i}_chunk_0", f"Paper {i}", 0.9 - i * 0.1) for i in range(5)],
        }
        mock_opensearch_client.search_chunks_hybrid = MagicMock(return_value=_topk_return)
        mock_opensearch_client.asearch_chunks_hybrid = AsyncMock(return_value=_topk_return)

        result = await multi_query_service.retrieve_with_multi_query("test query", top_k=3)

        assert len(result.results) <= 3

    @pytest.mark.asyncio
    async def test_retrieve_with_multi_query_empty_results(self, multi_query_service, mock_opensearch_client):
        """Verify returns empty MultiQueryResult when no hits."""
        mock_opensearch_client.search_chunks_hybrid = MagicMock(return_value={"total": 0, "hits": []})
        mock_opensearch_client.asearch_chunks_hybrid = AsyncMock(return_value={"total": 0, "hits": []})

        result = await multi_query_service.retrieve_with_multi_query("obscure topic")

        assert result.results == []
        assert result.original_query == "obscure topic"

    @pytest.mark.asyncio
    async def test_retrieve_with_multi_query_default_top_k(self, multi_query_service, mock_opensearch_client):
        """Verify default top_k is 20."""
        await multi_query_service.retrieve_with_multi_query("test query")

        # Each individual search should use top_k=20
        call_kwargs = mock_opensearch_client.asearch_chunks_hybrid.call_args
        assert call_kwargs.kwargs.get("size") == 20


class TestMultiQuerySettings:
    """Tests for FR-7: Configuration."""

    def test_multi_query_settings_defaults(self):
        """Verify default settings values."""
        from src.config import MultiQuerySettings

        settings = MultiQuerySettings()
        assert settings.enabled is True
        assert settings.num_queries == 3
        assert settings.temperature == 0.7
        assert settings.max_tokens == 300
        assert settings.rrf_k == 60

    def test_multi_query_settings_from_env(self, monkeypatch):
        """Verify settings loaded from environment variables."""
        from src.config import MultiQuerySettings

        monkeypatch.setenv("MULTI_QUERY__ENABLED", "false")
        monkeypatch.setenv("MULTI_QUERY__NUM_QUERIES", "5")
        monkeypatch.setenv("MULTI_QUERY__TEMPERATURE", "0.9")
        monkeypatch.setenv("MULTI_QUERY__MAX_TOKENS", "500")
        monkeypatch.setenv("MULTI_QUERY__RRF_K", "100")

        settings = MultiQuerySettings()
        assert settings.enabled is False
        assert settings.num_queries == 5
        assert settings.temperature == 0.9
        assert settings.max_tokens == 500
        assert settings.rrf_k == 100

    def test_multi_query_settings_in_root_settings(self):
        """Verify MultiQuerySettings is accessible from root Settings."""
        from src.config import Settings

        settings = Settings()
        assert hasattr(settings, "multi_query")
        assert settings.multi_query.enabled is True


class TestMultiQueryFactory:
    """Tests for FR-8: Factory function."""

    def test_create_multi_query_service(self):
        """Verify factory creates a MultiQueryService with correct dependencies."""
        from src.config import MultiQuerySettings
        from src.services.retrieval.factory import create_multi_query_service
        from src.services.retrieval.multi_query import MultiQueryService

        llm = AsyncMock()
        embeddings = AsyncMock()
        opensearch = AsyncMock()
        settings = MultiQuerySettings()

        service = create_multi_query_service(
            settings=settings,
            llm_provider=llm,
            embeddings_client=embeddings,
            opensearch_client=opensearch,
        )

        assert isinstance(service, MultiQueryService)
