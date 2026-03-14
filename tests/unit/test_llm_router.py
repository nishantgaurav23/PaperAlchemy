"""Tests for S5b.2: Multi-Provider LLM Routing.

Tests the LLMRouter class, TaskType enum, routing logic, fallback chain,
cost tracking, and factory/DI integration.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from src.services.llm.provider import HealthStatus, LLMResponse, UsageMetadata
from src.services.llm.router import LLMRouter, ProviderUsageStats, TaskType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def usage_metadata() -> UsageMetadata:
    return UsageMetadata(prompt_tokens=10, completion_tokens=20, total_tokens=30)


@pytest.fixture()
def _make_mock_provider():
    """Factory to create a mock LLMProvider with configurable provider name."""

    def _factory(name: str, *, fail: bool = False, usage: UsageMetadata | None = None):
        provider = AsyncMock()
        if fail:
            provider.generate.side_effect = Exception(f"{name} failed")
            provider.generate_stream.side_effect = Exception(f"{name} stream failed")
        else:
            resp = LLMResponse(
                text=f"response from {name}",
                model=f"{name}-model",
                provider=name,
                usage=usage,
            )
            provider.generate.return_value = resp

            async def _stream(*args, **kwargs):
                yield f"chunk from {name}"

            provider.generate_stream = MagicMock(side_effect=lambda *a, **kw: _stream(*a, **kw))
        provider.health_check.return_value = HealthStatus(
            healthy=not fail, provider=name, message=f"{name} ok" if not fail else f"{name} down"
        )
        provider.get_langchain_model.return_value = MagicMock(name=f"{name}_langchain")
        provider.close.return_value = None
        return provider

    return _factory


@pytest.fixture()
def mock_providers(_make_mock_provider, usage_metadata):
    """Return a dict of three mock providers: gemini, anthropic, ollama."""
    return {
        "gemini": _make_mock_provider("gemini", usage=usage_metadata),
        "anthropic": _make_mock_provider("anthropic", usage=usage_metadata),
        "ollama": _make_mock_provider("ollama", usage=usage_metadata),
    }


@pytest.fixture()
def default_routing_settings():
    from src.config import LLMRoutingSettings

    return LLMRoutingSettings()


@pytest.fixture()
def router(mock_providers, default_routing_settings):
    return LLMRouter(providers=mock_providers, routing_settings=default_routing_settings)


# ---------------------------------------------------------------------------
# TaskType Enum
# ---------------------------------------------------------------------------


class TestTaskType:
    def test_all_task_types_defined(self):
        expected = {"RESEARCH_QA", "CODE_GENERATION", "SUMMARIZATION", "GRADING", "QUERY_REWRITE", "GENERAL"}
        actual = {t.name for t in TaskType}
        assert expected == actual

    def test_task_type_values(self):
        assert TaskType.RESEARCH_QA.value == "research_qa"
        assert TaskType.CODE_GENERATION.value == "code_generation"
        assert TaskType.GENERAL.value == "general"


# ---------------------------------------------------------------------------
# Routing Logic
# ---------------------------------------------------------------------------


class TestRouting:
    def test_route_research_qa_to_gemini(self, router, mock_providers):
        provider = router.route(TaskType.RESEARCH_QA)
        assert provider is mock_providers["gemini"]

    def test_route_code_gen_to_anthropic(self, router, mock_providers):
        provider = router.route(TaskType.CODE_GENERATION)
        assert provider is mock_providers["anthropic"]

    def test_route_summarization_to_gemini(self, router, mock_providers):
        provider = router.route(TaskType.SUMMARIZATION)
        assert provider is mock_providers["gemini"]

    def test_route_grading_to_gemini(self, router, mock_providers):
        provider = router.route(TaskType.GRADING)
        assert provider is mock_providers["gemini"]

    def test_route_query_rewrite_to_gemini(self, router, mock_providers):
        provider = router.route(TaskType.QUERY_REWRITE)
        assert provider is mock_providers["gemini"]

    def test_route_general_to_default(self, router, mock_providers):
        provider = router.route(TaskType.GENERAL)
        assert provider is mock_providers["gemini"]

    def test_route_custom_mapping(self, mock_providers):
        """Custom routing: code gen -> ollama."""
        from src.config import LLMRoutingSettings

        settings = LLMRoutingSettings(code_generation_provider="ollama")
        r = LLMRouter(providers=mock_providers, routing_settings=settings)
        assert r.route(TaskType.CODE_GENERATION) is mock_providers["ollama"]

    def test_unknown_provider_falls_back_to_ollama(self, mock_providers):
        """If config references a provider that doesn't exist, fall back to ollama."""
        from src.config import LLMRoutingSettings

        settings = LLMRoutingSettings(research_qa_provider="nonexistent")
        r = LLMRouter(providers=mock_providers, routing_settings=settings)
        provider = r.route(TaskType.RESEARCH_QA)
        assert provider is mock_providers["ollama"]


# ---------------------------------------------------------------------------
# generate() delegation
# ---------------------------------------------------------------------------


class TestGenerate:
    @pytest.mark.asyncio
    async def test_generate_delegates_to_correct_provider(self, router, mock_providers):
        result = await router.generate("test prompt", task_type=TaskType.RESEARCH_QA)
        assert result.provider == "gemini"
        mock_providers["gemini"].generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_generate_code_gen_delegates_to_anthropic(self, router, mock_providers):
        result = await router.generate("write code", task_type=TaskType.CODE_GENERATION)
        assert result.provider == "anthropic"
        mock_providers["anthropic"].generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_generate_default_task_type_is_general(self, router, mock_providers):
        result = await router.generate("hello")
        assert result.provider == "gemini"  # general -> gemini (default)

    @pytest.mark.asyncio
    async def test_generate_passes_kwargs(self, router, mock_providers):
        await router.generate("p", task_type=TaskType.RESEARCH_QA, temperature=0.5, max_tokens=100)
        call_kwargs = mock_providers["gemini"].generate.call_args
        assert call_kwargs.kwargs["temperature"] == 0.5
        assert call_kwargs.kwargs["max_tokens"] == 100


# ---------------------------------------------------------------------------
# generate_stream() delegation
# ---------------------------------------------------------------------------


class TestGenerateStream:
    @pytest.mark.asyncio
    async def test_generate_stream_delegates_to_correct_provider(self, router):
        chunks = []
        async for chunk in router.generate_stream("test", task_type=TaskType.RESEARCH_QA):
            chunks.append(chunk)
        assert chunks == ["chunk from gemini"]

    @pytest.mark.asyncio
    async def test_generate_stream_code_gen(self, router):
        chunks = []
        async for chunk in router.generate_stream("test", task_type=TaskType.CODE_GENERATION):
            chunks.append(chunk)
        assert chunks == ["chunk from anthropic"]


# ---------------------------------------------------------------------------
# Fallback Chain
# ---------------------------------------------------------------------------


class TestFallback:
    @pytest.mark.asyncio
    async def test_fallback_on_primary_failure(self, _make_mock_provider, usage_metadata):
        from src.config import LLMRoutingSettings

        providers = {
            "gemini": _make_mock_provider("gemini", fail=True),
            "anthropic": _make_mock_provider("anthropic", usage=usage_metadata),
            "ollama": _make_mock_provider("ollama", usage=usage_metadata),
        }
        settings = LLMRoutingSettings(fallback_enabled=True, fallback_order="gemini,anthropic,ollama")
        r = LLMRouter(providers=providers, routing_settings=settings)

        result = await r.generate("test", task_type=TaskType.RESEARCH_QA)
        # gemini fails -> anthropic is next in fallback
        assert result.provider == "anthropic"

    @pytest.mark.asyncio
    async def test_fallback_chain_exhausted(self, _make_mock_provider):
        from src.config import LLMRoutingSettings
        from src.exceptions import LLMServiceError

        providers = {
            "gemini": _make_mock_provider("gemini", fail=True),
            "anthropic": _make_mock_provider("anthropic", fail=True),
            "ollama": _make_mock_provider("ollama", fail=True),
        }
        settings = LLMRoutingSettings(fallback_enabled=True)
        r = LLMRouter(providers=providers, routing_settings=settings)

        with pytest.raises(LLMServiceError, match="All LLM providers failed"):
            await r.generate("test", task_type=TaskType.RESEARCH_QA)

    @pytest.mark.asyncio
    async def test_fallback_disabled(self, _make_mock_provider):
        from src.config import LLMRoutingSettings
        from src.exceptions import LLMServiceError

        providers = {
            "gemini": _make_mock_provider("gemini", fail=True),
            "anthropic": _make_mock_provider("anthropic"),
            "ollama": _make_mock_provider("ollama"),
        }
        settings = LLMRoutingSettings(fallback_enabled=False)
        r = LLMRouter(providers=providers, routing_settings=settings)

        with pytest.raises(LLMServiceError):
            await r.generate("test", task_type=TaskType.RESEARCH_QA)

    @pytest.mark.asyncio
    async def test_fallback_skips_to_second_when_first_also_fails(self, _make_mock_provider, usage_metadata):
        from src.config import LLMRoutingSettings

        providers = {
            "gemini": _make_mock_provider("gemini", fail=True),
            "anthropic": _make_mock_provider("anthropic", fail=True),
            "ollama": _make_mock_provider("ollama", usage=usage_metadata),
        }
        settings = LLMRoutingSettings(fallback_enabled=True, fallback_order="gemini,anthropic,ollama")
        r = LLMRouter(providers=providers, routing_settings=settings)

        result = await r.generate("test", task_type=TaskType.RESEARCH_QA)
        assert result.provider == "ollama"


# ---------------------------------------------------------------------------
# Cost Tracking
# ---------------------------------------------------------------------------


class TestCostTracking:
    @pytest.mark.asyncio
    async def test_cost_tracking_increments(self, router):
        await router.generate("test", task_type=TaskType.RESEARCH_QA)
        stats = router.get_usage_stats()
        assert stats["gemini"].total_requests == 1
        assert stats["gemini"].total_prompt_tokens == 10
        assert stats["gemini"].total_completion_tokens == 20
        assert stats["gemini"].total_tokens == 30

    @pytest.mark.asyncio
    async def test_cost_tracking_multiple_calls(self, router):
        await router.generate("q1", task_type=TaskType.RESEARCH_QA)
        await router.generate("q2", task_type=TaskType.RESEARCH_QA)
        stats = router.get_usage_stats()
        assert stats["gemini"].total_requests == 2
        assert stats["gemini"].total_tokens == 60

    @pytest.mark.asyncio
    async def test_cost_tracking_reset(self, router):
        await router.generate("test", task_type=TaskType.RESEARCH_QA)
        router.reset_usage_stats()
        stats = router.get_usage_stats()
        assert stats["gemini"].total_requests == 0
        assert stats["gemini"].total_tokens == 0

    @pytest.mark.asyncio
    async def test_cost_tracking_failed_requests(self, _make_mock_provider):
        from src.config import LLMRoutingSettings

        providers = {
            "gemini": _make_mock_provider("gemini", fail=True),
            "ollama": _make_mock_provider("ollama"),
        }
        settings = LLMRoutingSettings(fallback_enabled=True, fallback_order="gemini,ollama")
        r = LLMRouter(providers=providers, routing_settings=settings)

        await r.generate("test", task_type=TaskType.RESEARCH_QA)
        stats = r.get_usage_stats()
        assert stats["gemini"].failed_requests == 1

    @pytest.mark.asyncio
    async def test_cost_tracking_no_usage_metadata(self, _make_mock_provider):
        """Provider returns no usage metadata -> skip token counting."""
        from src.config import LLMRoutingSettings

        providers = {
            "gemini": _make_mock_provider("gemini", usage=None),
            "ollama": _make_mock_provider("ollama"),
        }
        settings = LLMRoutingSettings()
        r = LLMRouter(providers=providers, routing_settings=settings)

        await r.generate("test", task_type=TaskType.RESEARCH_QA)
        stats = r.get_usage_stats()
        assert stats["gemini"].total_requests == 1
        assert stats["gemini"].total_tokens == 0  # no usage data


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check_aggregates_all_providers(self, router):
        result = await router.health_check()
        assert result.healthy is True
        assert result.provider == "router"
        assert "gemini" in result.message
        assert "anthropic" in result.message
        assert "ollama" in result.message

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_when_any_fails(self, _make_mock_provider, usage_metadata):
        from src.config import LLMRoutingSettings

        providers = {
            "gemini": _make_mock_provider("gemini", fail=True),
            "anthropic": _make_mock_provider("anthropic", usage=usage_metadata),
        }
        r = LLMRouter(providers=providers, routing_settings=LLMRoutingSettings())
        result = await r.health_check()
        # Still healthy overall if at least one provider works
        assert result.healthy is True


# ---------------------------------------------------------------------------
# get_langchain_model
# ---------------------------------------------------------------------------


class TestGetLangchainModel:
    def test_get_langchain_model_routes_by_task(self, router, mock_providers):
        model = router.get_langchain_model(task_type=TaskType.CODE_GENERATION)
        mock_providers["anthropic"].get_langchain_model.assert_called_once()
        assert model is not None

    def test_get_langchain_model_default_general(self, router, mock_providers):
        router.get_langchain_model()
        mock_providers["gemini"].get_langchain_model.assert_called_once()


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


class TestClose:
    @pytest.mark.asyncio
    async def test_close_closes_all_providers(self, router, mock_providers):
        await router.close()
        for p in mock_providers.values():
            p.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# LLMRoutingSettings
# ---------------------------------------------------------------------------


class TestLLMRoutingSettings:
    def test_default_values(self):
        from src.config import LLMRoutingSettings

        s = LLMRoutingSettings()
        assert s.research_qa_provider == "gemini"
        assert s.code_generation_provider == "anthropic"
        assert s.summarization_provider == "gemini"
        assert s.grading_provider == "gemini"
        assert s.query_rewrite_provider == "gemini"
        assert s.general_provider == "gemini"
        assert s.fallback_enabled is True
        assert s.fallback_order == "gemini,anthropic,ollama"
        assert s.cost_tracking_enabled is True

    def test_from_env(self, monkeypatch):
        from src.config import LLMRoutingSettings

        monkeypatch.setenv("LLM_ROUTING__RESEARCH_QA_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_ROUTING__FALLBACK_ENABLED", "false")
        s = LLMRoutingSettings()
        assert s.research_qa_provider == "ollama"
        assert s.fallback_enabled is False


# ---------------------------------------------------------------------------
# ProviderUsageStats
# ---------------------------------------------------------------------------


class TestProviderUsageStats:
    def test_default_values(self):
        stats = ProviderUsageStats(provider_name="gemini")
        assert stats.total_requests == 0
        assert stats.total_prompt_tokens == 0
        assert stats.total_completion_tokens == 0
        assert stats.total_tokens == 0
        assert stats.failed_requests == 0


# ---------------------------------------------------------------------------
# Factory / DI Integration
# ---------------------------------------------------------------------------


class TestFactory:
    def test_create_llm_router(self, monkeypatch):
        """create_llm_router() returns an LLMRouter instance."""
        from src.config import Settings
        from src.services.llm.factory import create_llm_router

        # Ensure no real API keys
        monkeypatch.delenv("GEMINI__API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC__API_KEY", raising=False)
        settings = Settings()
        router = create_llm_router(settings)
        assert isinstance(router, LLMRouter)
        # Should at least have ollama
        assert "ollama" in router._providers
