"""Tests for AgentContext dataclass and create_agent_context factory (FR-2, FR-5)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import fields

import pytest
from src.services.agents.context import AgentContext, create_agent_context
from src.services.llm.provider import HealthStatus, LLMProvider, LLMResponse


class MockLLMProvider:
    """Minimal LLMProvider implementation for testing."""

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return LLMResponse(text="mock", model="mock", provider="mock")

    async def generate_stream(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        yield "mock"

    async def health_check(self) -> HealthStatus:
        return HealthStatus(healthy=True, provider="mock", message="ok")

    def get_langchain_model(self, *, model: str | None = None, temperature: float | None = None):
        return None

    async def close(self) -> None:
        pass


class TestAgentContext:
    def test_is_dataclass(self):
        """AgentContext should be a dataclass."""
        assert hasattr(AgentContext, "__dataclass_fields__")

    def test_has_slots(self):
        """AgentContext should use slots for performance."""
        assert hasattr(AgentContext, "__slots__")

    def test_creation_with_provider(self):
        provider = MockLLMProvider()
        ctx = AgentContext(llm_provider=provider)
        assert ctx.llm_provider is provider

    def test_default_values(self):
        provider = MockLLMProvider()
        ctx = AgentContext(llm_provider=provider)
        assert ctx.retrieval_pipeline is None
        assert ctx.cache_service is None
        assert ctx.model_name == ""
        assert ctx.temperature == 0.7
        assert ctx.top_k == 5
        assert ctx.max_retrieval_attempts == 3
        assert ctx.guardrail_threshold == 40
        assert ctx.trace_id is None
        assert ctx.user_id == "api_user"

    def test_overrides(self):
        provider = MockLLMProvider()
        ctx = AgentContext(
            llm_provider=provider,
            model_name="gemini-3-flash",
            temperature=0.3,
            top_k=10,
            max_retrieval_attempts=5,
            guardrail_threshold=60,
            user_id="test_user",
            trace_id="trace-123",
        )
        assert ctx.model_name == "gemini-3-flash"
        assert ctx.temperature == 0.3
        assert ctx.top_k == 10
        assert ctx.max_retrieval_attempts == 5
        assert ctx.guardrail_threshold == 60
        assert ctx.user_id == "test_user"
        assert ctx.trace_id == "trace-123"

    def test_optional_services_none(self):
        provider = MockLLMProvider()
        ctx = AgentContext(llm_provider=provider)
        assert ctx.retrieval_pipeline is None
        assert ctx.cache_service is None

    def test_provider_satisfies_protocol(self):
        provider = MockLLMProvider()
        assert isinstance(provider, LLMProvider)

    def test_field_names(self):
        expected = {
            "llm_provider",
            "retrieval_pipeline",
            "cache_service",
            "web_search_service",
            "arxiv_client",
            "model_name",
            "temperature",
            "top_k",
            "max_retrieval_attempts",
            "guardrail_threshold",
            "trace_id",
            "user_id",
        }
        actual = {f.name for f in fields(AgentContext)}
        assert expected == actual


class TestCreateAgentContext:
    def test_factory_with_provider(self):
        provider = MockLLMProvider()
        ctx = create_agent_context(llm_provider=provider)
        assert ctx.llm_provider is provider
        assert ctx.temperature == 0.7

    def test_factory_with_overrides(self):
        provider = MockLLMProvider()
        ctx = create_agent_context(
            llm_provider=provider,
            temperature=0.5,
            top_k=8,
            user_id="custom_user",
        )
        assert ctx.temperature == 0.5
        assert ctx.top_k == 8
        assert ctx.user_id == "custom_user"

    def test_factory_requires_provider(self):
        # create_agent_context requires llm_provider kwarg
        with pytest.raises(TypeError):
            create_agent_context()  # type: ignore[call-arg]
