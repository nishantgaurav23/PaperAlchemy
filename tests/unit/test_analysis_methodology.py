"""Tests for the methodology & findings analysis service (S8.4).

Tests extraction of structured methodology analysis from papers, content preparation,
API endpoint, and error handling. All external services are mocked.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from src.exceptions import InsufficientContentError, LLMServiceError, PaperNotFoundError
from src.services.llm.provider import LLMResponse, UsageMetadata

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PAPER_ID = uuid.UUID("12345678-1234-5678-1234-567812345678")

MOCK_METHODOLOGY_JSON = json.dumps(
    {
        "research_design": (
            "Experimental study comparing attention-based architectures "
            "against recurrent baselines on machine translation tasks."
        ),
        "datasets": [
            {"name": "WMT 2014 EN-DE", "description": "EN-DE translation benchmark", "size": "4.5M pairs"},
            {"name": "WMT 2014 EN-FR", "description": "EN-FR translation benchmark", "size": "36M pairs"},
        ],
        "baselines": [
            "GNMT (Google Neural Machine Translation)",
            "ConvS2S (Convolutional Sequence to Sequence)",
            "ByteNet",
        ],
        "key_results": [
            {"metric": "BLEU", "value": "28.4", "context": "WMT 2014 EN-DE, new SOTA"},
            {"metric": "BLEU", "value": "41.0", "context": "WMT 2014 EN-FR, beats ensembles"},
            {"metric": "Training cost", "value": "3.5 days on 8 P100 GPUs", "context": "Fraction of cost"},
        ],
        "statistical_significance": "Results exceed previous SOTA by >2 BLEU points.",
        "reproducibility_notes": (
            "Architecture and hyperparameters fully specified. "
            "Training on 8 NVIDIA P100 GPUs. Code not released."
        ),
    }
)

MOCK_THEORETICAL_JSON = json.dumps(
    {
        "research_design": "Theoretical analysis of self-attention mechanisms with mathematical proofs of expressiveness.",
        "datasets": [],
        "baselines": [],
        "key_results": [],
        "statistical_significance": None,
        "reproducibility_notes": "Proofs provided in appendix. No implementation required.",
    }
)


def _make_paper_orm(
    abstract: str = "We propose a new architecture, the Transformer, based entirely on attention mechanisms.",
    sections: list | None = None,
    pdf_content: str | None = "Full text about transformer methodology and experimental results.",
) -> MagicMock:
    paper = MagicMock()
    paper.id = PAPER_ID
    paper.title = "Attention Is All You Need"
    paper.authors = ["Vaswani et al."]
    paper.abstract = abstract
    paper.sections = (
        sections
        if sections is not None
        else [
            {"title": "Abstract", "content": "We propose a new architecture.", "level": 1},
            {"title": "Introduction", "content": "Sequence modeling has been dominated by RNNs.", "level": 1},
            {"title": "Methodology", "content": "We use multi-head attention with scaled dot-product.", "level": 1},
            {"title": "Experiments", "content": "We train on WMT 2014 EN-DE and EN-FR benchmarks.", "level": 1},
            {"title": "Results", "content": "Our model achieves 28.4 BLEU on WMT 2014 EN-DE.", "level": 1},
            {"title": "Discussion", "content": "The Transformer outperforms all previous models.", "level": 1},
            {"title": "Conclusion", "content": "Attention-only models are viable for sequence transduction.", "level": 1},
        ]
    )
    paper.pdf_content = pdf_content
    return paper


def _make_llm_response(text: str = MOCK_METHODOLOGY_JSON) -> LLMResponse:
    return LLMResponse(
        text=text,
        model="gemini-3-flash",
        provider="gemini",
        usage=UsageMetadata(prompt_tokens=600, completion_tokens=300, total_tokens=900, latency_ms=1100.0),
    )


@pytest.fixture
def mock_paper_repo():
    repo = AsyncMock()
    repo.get_by_id = AsyncMock(return_value=_make_paper_orm())
    return repo


@pytest.fixture
def mock_llm_provider():
    provider = AsyncMock()
    provider.generate = AsyncMock(return_value=_make_llm_response())
    return provider


# ---------------------------------------------------------------------------
# FR-1: Methodology & Findings Analysis
# ---------------------------------------------------------------------------


class TestAnalyzeMethodology:
    """Tests for MethodologyService.analyze_methodology()."""

    @pytest.mark.asyncio
    async def test_analyze_methodology_full_paper(self, mock_paper_repo, mock_llm_provider):
        """Full paper with sections returns structured analysis with all 6 fields."""
        from src.services.analysis.methodology import MethodologyService

        service = MethodologyService(llm_provider=mock_llm_provider, paper_repo=mock_paper_repo)
        result = await service.analyze_methodology(PAPER_ID)

        a = result.analysis
        assert a.research_design
        assert len(a.datasets) >= 1
        assert len(a.baselines) >= 1
        assert len(a.key_results) >= 1
        assert a.statistical_significance is not None
        assert a.reproducibility_notes is not None
        mock_llm_provider.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_methodology_abstract_only(self, mock_paper_repo, mock_llm_provider):
        """Paper with abstract but no sections generates analysis with warning."""
        from src.services.analysis.methodology import MethodologyService

        mock_paper_repo.get_by_id = AsyncMock(return_value=_make_paper_orm(sections=[], pdf_content=None))
        service = MethodologyService(llm_provider=mock_llm_provider, paper_repo=mock_paper_repo)
        result = await service.analyze_methodology(PAPER_ID)

        assert result.analysis.research_design
        assert result.warning is not None

    @pytest.mark.asyncio
    async def test_analyze_methodology_paper_not_found(self, mock_paper_repo, mock_llm_provider):
        """Non-existent paper_id raises PaperNotFoundError."""
        from src.services.analysis.methodology import MethodologyService

        mock_paper_repo.get_by_id = AsyncMock(return_value=None)
        service = MethodologyService(llm_provider=mock_llm_provider, paper_repo=mock_paper_repo)

        with pytest.raises(PaperNotFoundError):
            await service.analyze_methodology(uuid.UUID("00000000-0000-0000-0000-000000000000"))

    @pytest.mark.asyncio
    async def test_analyze_methodology_insufficient_content(self, mock_paper_repo, mock_llm_provider):
        """Paper with no abstract and no sections raises InsufficientContentError."""
        from src.services.analysis.methodology import MethodologyService

        mock_paper_repo.get_by_id = AsyncMock(return_value=_make_paper_orm(abstract="", sections=[], pdf_content=None))
        service = MethodologyService(llm_provider=mock_llm_provider, paper_repo=mock_paper_repo)

        with pytest.raises(InsufficientContentError):
            await service.analyze_methodology(PAPER_ID)

    @pytest.mark.asyncio
    async def test_analyze_methodology_llm_failure(self, mock_paper_repo, mock_llm_provider):
        """LLM provider error raises LLMServiceError."""
        from src.services.analysis.methodology import MethodologyService

        mock_llm_provider.generate = AsyncMock(side_effect=Exception("Gemini API down"))
        service = MethodologyService(llm_provider=mock_llm_provider, paper_repo=mock_paper_repo)

        with pytest.raises(LLMServiceError):
            await service.analyze_methodology(PAPER_ID)

    @pytest.mark.asyncio
    async def test_analyze_methodology_theoretical_paper(self, mock_paper_repo, mock_llm_provider):
        """Theoretical paper -> empty datasets/results, research_design notes 'theoretical'."""
        from src.services.analysis.methodology import MethodologyService

        mock_llm_provider.generate = AsyncMock(return_value=_make_llm_response(text=MOCK_THEORETICAL_JSON))
        service = MethodologyService(llm_provider=mock_llm_provider, paper_repo=mock_paper_repo)
        result = await service.analyze_methodology(PAPER_ID)

        a = result.analysis
        assert "theoretical" in a.research_design.lower()
        assert len(a.datasets) == 0
        assert len(a.key_results) == 0

    @pytest.mark.asyncio
    async def test_analyze_methodology_malformed_llm_output(self, mock_paper_repo, mock_llm_provider):
        """Malformed LLM output falls back to raw extraction."""
        from src.services.analysis.methodology import MethodologyService

        mock_llm_provider.generate = AsyncMock(
            return_value=_make_llm_response(text="This is not JSON at all, just plain text about methodology.")
        )
        service = MethodologyService(llm_provider=mock_llm_provider, paper_repo=mock_paper_repo)
        result = await service.analyze_methodology(PAPER_ID)

        assert result.warning is not None


# ---------------------------------------------------------------------------
# FR-2: Content Preparation
# ---------------------------------------------------------------------------


class TestPrepareContent:
    """Tests for MethodologyService._prepare_content()."""

    def test_prepare_content_methodology_focus(self, mock_paper_repo, mock_llm_provider):
        """Content preparation prioritizes methodology/experiments/results sections."""
        from src.services.analysis.methodology import MethodologyService

        service = MethodologyService(llm_provider=mock_llm_provider, paper_repo=mock_paper_repo)
        paper = _make_paper_orm()
        content = service._prepare_content(paper)

        # Priority sections should appear before other sections
        assert "Methodology" in content or "methodology" in content.lower()
        assert "Results" in content or "results" in content.lower()
        assert "Experiments" in content or "experiments" in content.lower()
        assert len(content) > 0

    def test_prepare_content_abstract_only(self, mock_paper_repo, mock_llm_provider):
        """Paper with only abstract returns abstract content."""
        from src.services.analysis.methodology import MethodologyService

        service = MethodologyService(llm_provider=mock_llm_provider, paper_repo=mock_paper_repo)
        paper = _make_paper_orm(sections=[], pdf_content=None)
        content = service._prepare_content(paper)

        assert "transformer" in content.lower()

    def test_prepare_content_truncation(self, mock_paper_repo, mock_llm_provider):
        """Very long paper content is truncated to ~4000 words."""
        from src.services.analysis.methodology import MethodologyService

        service = MethodologyService(llm_provider=mock_llm_provider, paper_repo=mock_paper_repo)
        long_text = " ".join(["word"] * 10000)
        paper = _make_paper_orm(
            pdf_content=long_text,
            sections=[{"title": "Methods", "content": long_text, "level": 1}],
        )
        content = service._prepare_content(paper)

        word_count = len(content.split())
        assert word_count <= 4500  # ~4000 with some buffer for headers


# ---------------------------------------------------------------------------
# FR-3: Response Schema
# ---------------------------------------------------------------------------


class TestMethodologySchemas:
    """Tests for MethodologyAnalysis and MethodologyResponse Pydantic models."""

    def test_methodology_analysis_schema(self):
        """MethodologyAnalysis model validates with all 6 fields."""
        from src.schemas.api.analysis import DatasetInfo, MethodologyAnalysis, ResultEntry

        analysis = MethodologyAnalysis(
            research_design="Experimental study on machine translation.",
            datasets=[DatasetInfo(name="WMT 2014", description="Translation benchmark", size="4.5M pairs")],
            baselines=["GNMT", "ConvS2S"],
            key_results=[ResultEntry(metric="BLEU", value="28.4", context="WMT 2014 EN-DE")],
            statistical_significance="Exceeds SOTA by 2+ BLEU points",
            reproducibility_notes="8 P100 GPUs, 3.5 days training",
        )
        assert len(analysis.datasets) == 1
        assert analysis.datasets[0].name == "WMT 2014"
        assert len(analysis.key_results) == 1
        assert analysis.key_results[0].metric == "BLEU"

    def test_methodology_analysis_empty_fields(self):
        """Theoretical paper: datasets, baselines, key_results can be empty."""
        from src.schemas.api.analysis import MethodologyAnalysis

        analysis = MethodologyAnalysis(
            research_design="Theoretical analysis.",
            datasets=[],
            baselines=[],
            key_results=[],
            statistical_significance=None,
            reproducibility_notes=None,
        )
        assert analysis.datasets == []
        assert analysis.statistical_significance is None

    def test_methodology_response_schema(self):
        """MethodologyResponse includes analysis + metadata."""
        from src.schemas.api.analysis import MethodologyAnalysis, MethodologyResponse

        analysis = MethodologyAnalysis(
            research_design="Experimental study.",
            datasets=[],
            baselines=["Baseline A"],
            key_results=[],
        )
        response = MethodologyResponse(
            paper_id=PAPER_ID,
            analysis=analysis,
            model="gemini-3-flash",
            provider="gemini",
            latency_ms=1100.0,
        )
        assert response.paper_id == PAPER_ID
        assert response.model == "gemini-3-flash"
        assert response.latency_ms == 1100.0


# ---------------------------------------------------------------------------
# FR-3: API Endpoint
# ---------------------------------------------------------------------------


class TestMethodologyEndpoint:
    """Tests for POST /api/v1/papers/{paper_id}/methodology endpoint."""

    @pytest.mark.asyncio
    async def test_methodology_endpoint_success(self, mock_paper_repo, mock_llm_provider):
        """POST returns 200 with valid methodology analysis."""
        from httpx import ASGITransport, AsyncClient
        from src.dependency import get_llm_provider, get_paper_repository
        from src.main import create_app

        app = create_app()
        app.dependency_overrides[get_paper_repository] = lambda: mock_paper_repo
        app.dependency_overrides[get_llm_provider] = lambda: mock_llm_provider

        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.post(f"/api/v1/papers/{PAPER_ID}/methodology")

            assert resp.status_code == 200
            data = resp.json()
            assert "analysis" in data
            assert "research_design" in data["analysis"]
            assert "datasets" in data["analysis"]
            assert "key_results" in data["analysis"]
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_methodology_endpoint_not_found(self, mock_paper_repo, mock_llm_provider):
        """POST returns 404 for non-existent paper."""
        from httpx import ASGITransport, AsyncClient
        from src.dependency import get_llm_provider, get_paper_repository
        from src.main import create_app

        app = create_app()
        mock_paper_repo.get_by_id = AsyncMock(return_value=None)
        app.dependency_overrides[get_paper_repository] = lambda: mock_paper_repo
        app.dependency_overrides[get_llm_provider] = lambda: mock_llm_provider

        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.post(f"/api/v1/papers/{PAPER_ID}/methodology")

            assert resp.status_code == 404
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_force_regeneration(self, mock_paper_repo, mock_llm_provider):
        """?force=true calls LLM even if result could be cached."""
        from src.services.analysis.methodology import MethodologyService

        service = MethodologyService(llm_provider=mock_llm_provider, paper_repo=mock_paper_repo)

        await service.analyze_methodology(PAPER_ID, force=True)
        await service.analyze_methodology(PAPER_ID, force=True)

        assert mock_llm_provider.generate.call_count == 2
