"""Unit tests for HyDE (Hypothetical Document Embeddings) retrieval service.

Tests cover:
- Hypothetical document generation via LLM
- Embedding of hypothetical documents
- Full HyDE retrieval flow (generate -> embed -> search)
- Graceful fallback on LLM/embedding failures
- Edge cases (empty query, empty results, top_k)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from src.services.llm.provider import LLMResponse, UsageMetadata


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider that returns a hypothetical passage."""
    provider = AsyncMock()
    provider.generate = AsyncMock(
        return_value=LLMResponse(
            text=(
                "Transformer architectures have revolutionized natural language processing "
                "by replacing recurrence with self-attention mechanisms. The multi-head "
                "attention allows the model to jointly attend to information from different "
                "representation subspaces at different positions, enabling parallel processing "
                "of sequences and capturing long-range dependencies more effectively than "
                "traditional recurrent neural networks."
            ),
            model="gemini-2.0-flash",
            provider="gemini",
            usage=UsageMetadata(prompt_tokens=50, completion_tokens=80, total_tokens=130),
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
    """Mock OpenSearch client that returns fake search results."""
    client = MagicMock()
    _vector_results = {
            "total": 2,
            "hits": [
                {
                    "arxiv_id": "1706.03762",
                    "title": "Attention Is All You Need",
                    "authors": ["Vaswani", "Shazeer", "Parmar"],
                    "abstract": "The dominant sequence transduction models...",
                    "pdf_url": "https://arxiv.org/pdf/1706.03762",
                    "chunk_text": "We propose a new simple network architecture...",
                    "chunk_id": "1706.03762_chunk_0",
                    "section_title": "Introduction",
                    "score": 0.95,
                    "highlights": {},
                },
                {
                    "arxiv_id": "1810.04805",
                    "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                    "authors": ["Devlin", "Chang", "Lee", "Toutanova"],
                    "abstract": "We introduce a new language representation model...",
                    "pdf_url": "https://arxiv.org/pdf/1810.04805",
                    "chunk_text": "BERT is designed to pre-train deep bidirectional...",
                    "chunk_id": "1810.04805_chunk_0",
                    "section_title": "Model Architecture",
                    "score": 0.88,
                    "highlights": {},
                },
            ],
        }
    client.search_chunks_vectors = MagicMock(return_value=_vector_results)
    client.asearch_chunks_vectors = AsyncMock(return_value=_vector_results)
    return client


@pytest.fixture
def hyde_settings():
    """HyDE settings for testing."""
    from src.config import HyDESettings

    return HyDESettings(enabled=True, max_tokens=300, temperature=0.3, timeout=30)


@pytest.fixture
def hyde_service(hyde_settings, mock_llm_provider, mock_embeddings_client, mock_opensearch_client):
    """Fully constructed HyDEService with mocked dependencies."""
    from src.services.retrieval.hyde import HyDEService

    return HyDEService(
        settings=hyde_settings,
        llm_provider=mock_llm_provider,
        embeddings_client=mock_embeddings_client,
        opensearch_client=mock_opensearch_client,
    )


class TestGenerateHypotheticalDocument:
    """Tests for FR-2: Hypothetical document generation."""

    @pytest.mark.asyncio
    async def test_generate_hypothetical_document(self, hyde_service, mock_llm_provider):
        """Verify LLM called with correct prompt, returns hypothetical text."""
        result = await hyde_service.generate_hypothetical_document("What are transformers in NLP?")

        assert isinstance(result, str)
        assert len(result) > 0
        mock_llm_provider.generate.assert_called_once()

        call_kwargs = mock_llm_provider.generate.call_args
        assert "What are transformers in NLP?" in call_kwargs[0][0] or "What are transformers in NLP?" in str(call_kwargs)

    @pytest.mark.asyncio
    async def test_generate_hypothetical_document_empty_query(self, hyde_service):
        """Verify ValueError raised for empty query."""
        with pytest.raises(ValueError, match="[Qq]uery.*empty|[Ee]mpty.*query"):
            await hyde_service.generate_hypothetical_document("")

    @pytest.mark.asyncio
    async def test_generate_hypothetical_document_whitespace_query(self, hyde_service):
        """Verify ValueError raised for whitespace-only query."""
        with pytest.raises(ValueError, match="[Qq]uery.*empty|[Ee]mpty.*query"):
            await hyde_service.generate_hypothetical_document("   ")

    @pytest.mark.asyncio
    async def test_generate_hypothetical_document_llm_failure(self, hyde_service, mock_llm_provider):
        """Verify fallback returns original query when LLM fails."""
        mock_llm_provider.generate.side_effect = Exception("LLM service unavailable")

        result = await hyde_service.generate_hypothetical_document("What are transformers?")

        assert result == "What are transformers?"

    @pytest.mark.asyncio
    async def test_generate_hypothetical_document_llm_empty_response(self, hyde_service, mock_llm_provider):
        """Verify fallback when LLM returns empty text."""
        mock_llm_provider.generate.return_value = LLMResponse(text="", model="gemini-2.0-flash", provider="gemini")

        result = await hyde_service.generate_hypothetical_document("What are transformers?")

        assert result == "What are transformers?"


class TestHyDEPrompt:
    """Tests for FR-2: Prompt format verification."""

    @pytest.mark.asyncio
    async def test_hyde_prompt_format(self, hyde_service, mock_llm_provider):
        """Verify the system prompt is academic/research-focused."""
        await hyde_service.generate_hypothetical_document("What is attention mechanism?")

        prompt = mock_llm_provider.generate.call_args[0][0]
        # Prompt should mention academic/research context
        prompt_lower = prompt.lower()
        assert any(word in prompt_lower for word in ["academic", "research", "paper", "passage", "scientific"]), (
            f"Prompt should be academic-focused, got: {prompt}"
        )

    @pytest.mark.asyncio
    async def test_hyde_uses_low_temperature(self, hyde_service, mock_llm_provider):
        """Verify temperature=0.3 is used for generation."""
        await hyde_service.generate_hypothetical_document("What is attention mechanism?")

        call_kwargs = mock_llm_provider.generate.call_args[1]
        assert call_kwargs.get("temperature") == 0.3

    @pytest.mark.asyncio
    async def test_hyde_uses_max_tokens(self, hyde_service, mock_llm_provider):
        """Verify max_tokens from settings is passed to LLM."""
        await hyde_service.generate_hypothetical_document("What is attention mechanism?")

        call_kwargs = mock_llm_provider.generate.call_args[1]
        assert call_kwargs.get("max_tokens") == 300


class TestRetrieveWithHyDE:
    """Tests for FR-1, FR-3, FR-4: Full HyDE retrieval flow."""

    @pytest.mark.asyncio
    async def test_retrieve_with_hyde_full_flow(
        self, hyde_service, mock_llm_provider, mock_embeddings_client, mock_opensearch_client
    ):
        """Verify end-to-end: generate -> embed -> search -> return results."""
        from src.services.retrieval.hyde import HyDEResult

        result = await hyde_service.retrieve_with_hyde("What are transformers in NLP?")

        assert isinstance(result, HyDEResult)
        assert len(result.hypothetical_document) > 0
        assert len(result.query_embedding) == 1024
        assert len(result.results) == 2
        assert result.results[0].arxiv_id == "1706.03762"

        # Verify the pipeline: LLM generate -> embed hypothetical -> vector search
        mock_llm_provider.generate.assert_called_once()
        mock_embeddings_client.embed_query.assert_called_once()
        mock_opensearch_client.asearch_chunks_vectors.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retrieve_with_hyde_empty_query(self, hyde_service):
        """Verify ValueError for empty query in retrieve flow."""
        with pytest.raises(ValueError, match="[Qq]uery.*empty|[Ee]mpty.*query"):
            await hyde_service.retrieve_with_hyde("")

    @pytest.mark.asyncio
    async def test_retrieve_with_hyde_fallback_on_llm_error(
        self, hyde_service, mock_llm_provider, mock_embeddings_client, mock_opensearch_client
    ):
        """Verify falls back to standard embedding when LLM fails."""
        mock_llm_provider.generate.side_effect = Exception("LLM unavailable")

        result = await hyde_service.retrieve_with_hyde("What are transformers?")

        # Should still return results (fallback to original query embedding)
        assert len(result.results) == 2
        assert result.hypothetical_document == "What are transformers?"
        # embed_query should be called with original query as fallback
        mock_embeddings_client.embed_query.assert_called_once()
        mock_opensearch_client.asearch_chunks_vectors.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retrieve_with_hyde_fallback_on_embed_error(
        self, hyde_service, mock_llm_provider, mock_embeddings_client, mock_opensearch_client
    ):
        """Verify falls back when embedding of hypothetical doc fails."""
        # First call (embed hypothetical) fails, second call (embed original query) succeeds
        mock_embeddings_client.embed_query.side_effect = [
            Exception("Jina API error"),
            [0.2] * 1024,
        ]

        result = await hyde_service.retrieve_with_hyde("What are transformers?")

        assert len(result.results) == 2
        # embed_query called twice: once for hypothetical (fails), once for original (succeeds)
        assert mock_embeddings_client.embed_query.call_count == 2

    @pytest.mark.asyncio
    async def test_retrieve_with_hyde_empty_results(self, hyde_service, mock_opensearch_client):
        """Verify returns empty HyDEResult when no search hits."""
        mock_opensearch_client.asearch_chunks_vectors.return_value = {"total": 0, "hits": []}

        result = await hyde_service.retrieve_with_hyde("obscure topic with no papers")

        assert result.results == []

    @pytest.mark.asyncio
    async def test_retrieve_with_hyde_respects_top_k(self, hyde_service, mock_opensearch_client):
        """Verify top_k parameter passed to search."""
        await hyde_service.retrieve_with_hyde("What are transformers?", top_k=5)

        call_kwargs = mock_opensearch_client.asearch_chunks_vectors.call_args
        assert call_kwargs[1].get("size") == 5 or call_kwargs.kwargs.get("size") == 5

    @pytest.mark.asyncio
    async def test_retrieve_with_hyde_default_top_k(self, hyde_service, mock_opensearch_client):
        """Verify default top_k is 20."""
        await hyde_service.retrieve_with_hyde("What are transformers?")

        call_kwargs = mock_opensearch_client.asearch_chunks_vectors.call_args
        assert call_kwargs[1].get("size") == 20 or call_kwargs.kwargs.get("size") == 20


class TestHyDESettings:
    """Tests for FR-6: Configuration."""

    def test_hyde_settings_defaults(self):
        """Verify default settings values."""
        from src.config import HyDESettings

        settings = HyDESettings()
        assert settings.enabled is True
        assert settings.max_tokens == 300
        assert settings.temperature == 0.3
        assert settings.timeout == 30

    def test_hyde_settings_from_env(self, monkeypatch):
        """Verify settings loaded from environment variables."""
        from src.config import HyDESettings

        monkeypatch.setenv("HYDE__ENABLED", "false")
        monkeypatch.setenv("HYDE__MAX_TOKENS", "500")
        monkeypatch.setenv("HYDE__TEMPERATURE", "0.5")
        monkeypatch.setenv("HYDE__TIMEOUT", "60")

        settings = HyDESettings()
        assert settings.enabled is False
        assert settings.max_tokens == 500
        assert settings.temperature == 0.5
        assert settings.timeout == 60

    def test_hyde_settings_in_root_settings(self):
        """Verify HyDESettings is accessible from root Settings."""
        from src.config import Settings

        settings = Settings()
        assert hasattr(settings, "hyde")
        assert settings.hyde.enabled is True


class TestHyDEFactory:
    """Tests for FR-7: Factory function."""

    def test_create_hyde_service(self):
        """Verify factory creates a HyDEService with correct dependencies."""
        from src.config import HyDESettings
        from src.services.retrieval.factory import create_hyde_service
        from src.services.retrieval.hyde import HyDEService

        llm = AsyncMock()
        embeddings = AsyncMock()
        opensearch = AsyncMock()
        settings = HyDESettings()

        service = create_hyde_service(
            settings=settings,
            llm_provider=llm,
            embeddings_client=embeddings,
            opensearch_client=opensearch,
        )

        assert isinstance(service, HyDEService)
