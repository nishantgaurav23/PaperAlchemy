"""Unit tests for the cross-encoder re-ranking service (S4b.1)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from src.config import RerankerSettings
from src.exceptions import RerankerError
from src.schemas.api.search import SearchHit

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reranker_settings() -> RerankerSettings:
    return RerankerSettings(
        model="cross-encoder/ms-marco-MiniLM-L-12-v2",
        top_k=5,
        device="cpu",
        provider="local",
        cohere_api_key="",
    )


@pytest.fixture
def cohere_settings() -> RerankerSettings:
    return RerankerSettings(
        model="rerank-v3.5",
        top_k=5,
        device="cpu",
        provider="cohere",
        cohere_api_key="test-key",
    )


@pytest.fixture
def sample_documents() -> list[dict]:
    return [
        {"text": "Transformers use self-attention mechanisms.", "id": "doc1"},
        {"text": "Convolutional networks excel at image classification.", "id": "doc2"},
        {"text": "Recurrent networks process sequences step by step.", "id": "doc3"},
        {"text": "BERT is a bidirectional transformer model.", "id": "doc4"},
        {"text": "GPT generates text autoregressively.", "id": "doc5"},
        {"text": "Attention is all you need for translation.", "id": "doc6"},
    ]


@pytest.fixture
def sample_search_hits() -> list[SearchHit]:
    return [
        SearchHit(arxiv_id="2301.001", title="Paper A", chunk_text="Transformers use attention.", score=0.8),
        SearchHit(arxiv_id="2301.002", title="Paper B", chunk_text="CNNs for images.", score=0.7),
        SearchHit(arxiv_id="2301.003", title="Paper C", chunk_text="", abstract="RNNs process sequences.", score=0.6),
        SearchHit(arxiv_id="2301.004", title="Paper D", chunk_text="", abstract="", score=0.5),
        SearchHit(arxiv_id="2301.005", title="Paper E", chunk_text="BERT pretraining.", score=0.4),
    ]


def _make_mock_cross_encoder():
    """Create a mock CrossEncoder that returns deterministic scores."""
    mock_model = MagicMock()
    # predict() returns raw logit scores — higher = more relevant
    mock_model.predict.return_value = np.array([0.9, -0.5, 0.2, 1.5, -1.0, 0.7])
    return mock_model


# ---------------------------------------------------------------------------
# FR-1: RerankerService Interface
# ---------------------------------------------------------------------------


class TestRerankInterface:
    @pytest.mark.asyncio
    async def test_rerank_returns_sorted_results(self, reranker_settings, sample_documents):
        """Results should be sorted by score descending."""
        from src.services.reranking.service import RerankerService

        mock_model = _make_mock_cross_encoder()
        service = RerankerService(settings=reranker_settings, model=mock_model)

        results = await service.rerank("transformer architecture", sample_documents)

        scores = [r.relevance_score for r in results]
        assert scores == sorted(scores, reverse=True), "Results not sorted by score descending"

    @pytest.mark.asyncio
    async def test_rerank_respects_top_k(self, reranker_settings, sample_documents):
        """Should return at most top_k results."""
        from src.services.reranking.service import RerankerService

        mock_model = _make_mock_cross_encoder()
        service = RerankerService(settings=reranker_settings, model=mock_model)

        results = await service.rerank("transformer", sample_documents, top_k=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_rerank_empty_documents(self, reranker_settings):
        """Empty document list should return empty list."""
        from src.services.reranking.service import RerankerService

        mock_model = _make_mock_cross_encoder()
        service = RerankerService(settings=reranker_settings, model=mock_model)

        results = await service.rerank("anything", [])
        assert results == []

    @pytest.mark.asyncio
    async def test_rerank_top_k_exceeds_docs(self, reranker_settings):
        """When top_k > len(docs), return all docs sorted."""
        from src.services.reranking.service import RerankerService

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 0.9])
        service = RerankerService(settings=reranker_settings, model=mock_model)

        docs = [
            {"text": "Doc one", "id": "a"},
            {"text": "Doc two", "id": "b"},
        ]
        results = await service.rerank("query", docs, top_k=10)
        assert len(results) == 2
        assert results[0].relevance_score >= results[1].relevance_score

    @pytest.mark.asyncio
    async def test_rerank_uses_default_top_k(self, reranker_settings, sample_documents):
        """When top_k not passed, use settings.top_k (5)."""
        from src.services.reranking.service import RerankerService

        mock_model = _make_mock_cross_encoder()
        service = RerankerService(settings=reranker_settings, model=mock_model)

        results = await service.rerank("query", sample_documents)
        assert len(results) == reranker_settings.top_k


# ---------------------------------------------------------------------------
# FR-2: Local Cross-Encoder Scoring
# ---------------------------------------------------------------------------


class TestLocalCrossEncoder:
    @pytest.mark.asyncio
    async def test_scores_normalized(self, reranker_settings, sample_documents):
        """All scores should be in 0.0 to 1.0 range (sigmoid normalization)."""
        from src.services.reranking.service import RerankerService

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([3.0, -3.0, 0.0, 1.0, -1.0, 0.5])
        service = RerankerService(settings=reranker_settings, model=mock_model)

        results = await service.rerank("query", sample_documents, top_k=6)
        for r in results:
            assert 0.0 <= r.relevance_score <= 1.0, f"Score {r.relevance_score} out of range"

    @pytest.mark.asyncio
    async def test_predict_called_with_pairs(self, reranker_settings):
        """CrossEncoder.predict() should be called with (query, doc_text) pairs."""
        from src.services.reranking.service import RerankerService

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 0.3])
        service = RerankerService(settings=reranker_settings, model=mock_model)

        docs = [{"text": "hello", "id": "1"}, {"text": "world", "id": "2"}]
        await service.rerank("test query", docs)

        call_args = mock_model.predict.call_args[0][0]
        assert call_args == [["test query", "hello"], ["test query", "world"]]

    @pytest.mark.asyncio
    async def test_empty_text_gets_zero_score(self, reranker_settings):
        """Documents with empty text should get score 0.0."""
        from src.services.reranking.service import RerankerService

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.8])
        service = RerankerService(settings=reranker_settings, model=mock_model)

        docs = [
            {"text": "real content", "id": "1"},
            {"text": "", "id": "2"},
        ]
        results = await service.rerank("query", docs, top_k=2)

        # The empty-text doc should have score 0.0
        empty_result = [r for r in results if r.index == 1][0]
        assert empty_result.relevance_score == 0.0

    @pytest.mark.asyncio
    async def test_rerank_result_fields(self, reranker_settings):
        """RerankResult should have index, relevance_score, and document fields."""
        from src.services.reranking.service import RerankerService

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5])
        service = RerankerService(settings=reranker_settings, model=mock_model)

        docs = [{"text": "hello world", "id": "doc1"}]
        results = await service.rerank("query", docs, top_k=1)

        assert len(results) == 1
        r = results[0]
        assert r.index == 0
        assert isinstance(r.relevance_score, float)
        assert r.document == docs[0]


# ---------------------------------------------------------------------------
# FR-4: Integration with SearchHit
# ---------------------------------------------------------------------------


class TestRerankSearchHits:
    @pytest.mark.asyncio
    async def test_rerank_search_hits(self, reranker_settings, sample_search_hits):
        """SearchHit objects should be re-ranked by cross-encoder scores."""
        from src.services.reranking.service import RerankerService

        mock_model = MagicMock()
        # 5 hits → all scorable (text extracted from chunk_text, abstract, or title)
        # Hits: chunk="Transformers...", chunk="CNNs...", abstract="RNNs...", title="Paper D", chunk="BERT..."
        mock_model.predict.return_value = np.array([0.3, 0.9, 0.1, 0.2, 0.7])
        service = RerankerService(settings=reranker_settings, model=mock_model)

        results = await service.rerank_search_hits("query", sample_search_hits, top_k=3)

        assert len(results) == 3
        assert all(isinstance(h, SearchHit) for h in results)
        # Scores should be descending
        scores = [h.score for h in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_rerank_search_hits_text_extraction(self, reranker_settings):
        """Text extraction priority: chunk_text > abstract > title."""
        from src.services.reranking.service import RerankerService

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 0.5, 0.5])
        service = RerankerService(settings=reranker_settings, model=mock_model)

        hits = [
            SearchHit(arxiv_id="1", title="Title A", chunk_text="Chunk text", abstract="Abstract text"),
            SearchHit(arxiv_id="2", title="Title B", chunk_text="", abstract="Abstract only"),
            SearchHit(arxiv_id="3", title="Title C", chunk_text="", abstract=""),
        ]
        await service.rerank_search_hits("query", hits, top_k=3)

        call_args = mock_model.predict.call_args[0][0]
        assert call_args[0][1] == "Chunk text"
        assert call_args[1][1] == "Abstract only"
        assert call_args[2][1] == "Title C"

    @pytest.mark.asyncio
    async def test_rerank_search_hits_no_text(self, reranker_settings):
        """Hits with no text (empty chunk_text, abstract, and title) get score 0.0."""
        from src.services.reranking.service import RerankerService

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9])
        service = RerankerService(settings=reranker_settings, model=mock_model)

        hits = [
            SearchHit(arxiv_id="1", title="Good Paper", chunk_text="Some content"),
            SearchHit(arxiv_id="2", title="", chunk_text="", abstract=""),
        ]
        results = await service.rerank_search_hits("query", hits, top_k=2)

        # The no-text hit should be last with score 0.0
        assert results[-1].arxiv_id == "2"
        assert results[-1].score == 0.0

    @pytest.mark.asyncio
    async def test_rerank_search_hits_empty_list(self, reranker_settings):
        """Empty hits list should return empty list."""
        from src.services.reranking.service import RerankerService

        mock_model = _make_mock_cross_encoder()
        service = RerankerService(settings=reranker_settings, model=mock_model)

        results = await service.rerank_search_hits("query", [], top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# FR-6: Factory Function
# ---------------------------------------------------------------------------


class TestFactory:
    def test_factory_creates_local_provider(self, reranker_settings):
        """Factory with provider='local' should create service with CrossEncoder model."""
        with patch("src.services.reranking.factory.CrossEncoder") as mock_ce:
            mock_instance = MagicMock()
            mock_ce.return_value = mock_instance

            from src.services.reranking.factory import create_reranker_service

            service = create_reranker_service(reranker_settings)

            mock_ce.assert_called_once_with(reranker_settings.model, device=reranker_settings.device)
            assert service is not None

    def test_factory_creates_cohere_provider(self, cohere_settings):
        """Factory with provider='cohere' should create service configured for Cohere."""
        from src.services.reranking.factory import create_reranker_service

        service = create_reranker_service(cohere_settings)
        assert service is not None
        assert service._provider == "cohere"

    def test_factory_invalid_provider(self):
        """Factory with unknown provider should raise RerankerError."""
        from src.services.reranking.factory import create_reranker_service

        settings = RerankerSettings(provider="unknown")
        with pytest.raises(RerankerError, match="Unknown reranker provider"):
            create_reranker_service(settings)


# ---------------------------------------------------------------------------
# FR-3: Cohere Rerank Provider
# ---------------------------------------------------------------------------


class TestCohereProvider:
    @pytest.mark.asyncio
    async def test_cohere_rerank(self, cohere_settings, sample_documents):
        """Cohere provider should call the API and return sorted results."""
        from src.services.reranking.service import RerankerService

        service = RerankerService(settings=cohere_settings, model=None, provider="cohere")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"index": 3, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.85},
                {"index": 5, "relevance_score": 0.70},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_async_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_async_client.return_value = mock_client

            results = await service.rerank("query", sample_documents, top_k=3)

        assert len(results) == 3
        scores = [r.relevance_score for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_cohere_api_failure(self, cohere_settings, sample_documents):
        """Cohere API failure should raise RerankerError."""
        from src.services.reranking.service import RerankerService

        service = RerankerService(settings=cohere_settings, model=None, provider="cohere")

        with patch("httpx.AsyncClient") as mock_async_client:
            mock_client = AsyncMock()
            mock_client.post.side_effect = Exception("API connection failed")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_async_client.return_value = mock_client

            with pytest.raises(RerankerError, match="Cohere rerank failed"):
                await service.rerank("query", sample_documents)
