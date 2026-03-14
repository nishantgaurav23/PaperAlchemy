"""Multi-provider LLM router — routes tasks to optimal providers.

Routes different task types to their optimal LLM provider:
- Research Q&A -> Gemini (knowledge-optimized)
- Code generation -> Claude (code-optimized)
- Local dev -> Ollama (no API costs)

Supports fallback chains and per-provider cost tracking.
"""

from __future__ import annotations

import enum
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

from src.config import LLMRoutingSettings
from src.exceptions import LLMServiceError
from src.services.llm.provider import HealthStatus, LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class TaskType(enum.Enum):
    """Task types for LLM routing decisions."""

    RESEARCH_QA = "research_qa"
    CODE_GENERATION = "code_generation"
    SUMMARIZATION = "summarization"
    GRADING = "grading"
    QUERY_REWRITE = "query_rewrite"
    GENERAL = "general"


@dataclass
class ProviderUsageStats:
    """Tracks per-provider token usage for cost estimation."""

    provider_name: str
    total_requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    failed_requests: int = 0


class LLMRouter:
    """Routes LLM calls to optimal providers based on task type.

    Implements the LLMProvider protocol so callers can use it transparently.
    """

    def __init__(self, providers: dict[str, LLMProvider], routing_settings: LLMRoutingSettings) -> None:
        self._providers = providers
        self._settings = routing_settings
        self._fallback_chain = [p.strip() for p in routing_settings.fallback_order.split(",") if p.strip()]
        self._usage_stats: dict[str, ProviderUsageStats] = {
            name: ProviderUsageStats(provider_name=name) for name in providers
        }

        # Build task -> provider-name mapping from settings
        self._task_mapping: dict[TaskType, str] = {
            TaskType.RESEARCH_QA: routing_settings.research_qa_provider,
            TaskType.CODE_GENERATION: routing_settings.code_generation_provider,
            TaskType.SUMMARIZATION: routing_settings.summarization_provider,
            TaskType.GRADING: routing_settings.grading_provider,
            TaskType.QUERY_REWRITE: routing_settings.query_rewrite_provider,
            TaskType.GENERAL: routing_settings.general_provider,
        }

    def route(self, task_type: TaskType) -> LLMProvider:
        """Resolve the primary provider for a given task type."""
        provider_name = self._task_mapping.get(task_type, self._settings.general_provider)
        if provider_name in self._providers:
            return self._providers[provider_name]

        logger.warning("Provider '%s' not available, falling back to ollama", provider_name)
        if "ollama" in self._providers:
            return self._providers["ollama"]

        # Last resort: return first available provider
        return next(iter(self._providers.values()))

    def _resolve_provider_name(self, task_type: TaskType) -> str:
        """Get the provider name string for a task type."""
        name = self._task_mapping.get(task_type, self._settings.general_provider)
        if name in self._providers:
            return name
        if "ollama" in self._providers:
            return "ollama"
        return next(iter(self._providers))

    async def generate(
        self,
        prompt: str,
        *,
        task_type: TaskType = TaskType.GENERAL,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a response, routing to the optimal provider for the task."""
        if self._settings.fallback_enabled:
            return await self._generate_with_fallback(
                prompt, task_type=task_type, model=model, temperature=temperature, max_tokens=max_tokens
            )

        provider_name = self._resolve_provider_name(task_type)
        provider = self._providers[provider_name]
        try:
            result = await provider.generate(prompt, model=model, temperature=temperature, max_tokens=max_tokens)
            self._record_usage(provider_name, result)
            return result
        except Exception as exc:
            self._record_failure(provider_name)
            raise LLMServiceError(f"Provider '{provider_name}' failed: {exc}") from exc

    async def _generate_with_fallback(
        self,
        prompt: str,
        *,
        task_type: TaskType,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Try primary provider, then fallback chain on failure."""
        primary_name = self._resolve_provider_name(task_type)
        errors: list[str] = []

        # Build ordered list: primary first, then fallback chain (skip duplicates)
        attempt_order = [primary_name]
        for name in self._fallback_chain:
            if name not in attempt_order and name in self._providers:
                attempt_order.append(name)

        for name in attempt_order:
            provider = self._providers[name]
            try:
                result = await provider.generate(prompt, model=model, temperature=temperature, max_tokens=max_tokens)
                self._record_usage(name, result)
                if name != primary_name:
                    logger.warning("Fallback to '%s' succeeded (primary '%s' failed)", name, primary_name)
                return result
            except Exception as exc:
                self._record_failure(name)
                errors.append(f"{name}: {exc}")
                logger.warning("Provider '%s' failed: %s", name, exc)

        raise LLMServiceError(f"All LLM providers failed: {'; '.join(errors)}")

    async def generate_stream(
        self,
        prompt: str,
        *,
        task_type: TaskType = TaskType.GENERAL,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream a response from the optimal provider for the task."""
        provider_name = self._resolve_provider_name(task_type)
        provider = self._providers[provider_name]
        async for chunk in provider.generate_stream(prompt, model=model, temperature=temperature, max_tokens=max_tokens):
            yield chunk

    async def health_check(self) -> HealthStatus:
        """Aggregate health status across all providers."""
        statuses: list[str] = []
        any_healthy = False

        for name, provider in self._providers.items():
            status = await provider.health_check()
            state = "healthy" if status.healthy else "unhealthy"
            statuses.append(f"{name}={state}")
            if status.healthy:
                any_healthy = True

        message = f"Router: {', '.join(statuses)}"
        return HealthStatus(healthy=any_healthy, provider="router", message=message)

    def get_langchain_model(
        self,
        *,
        task_type: TaskType = TaskType.GENERAL,
        model: str | None = None,
        temperature: float | None = None,
    ):
        """Return the langchain model from the routed provider."""
        provider = self.route(task_type)
        return provider.get_langchain_model(model=model, temperature=temperature)

    async def close(self) -> None:
        """Close all providers."""
        for provider in self._providers.values():
            await provider.close()

    # -- Cost tracking -------------------------------------------------------

    def _record_usage(self, provider_name: str, response: LLMResponse) -> None:
        """Record token usage from a successful response."""
        if provider_name not in self._usage_stats:
            return
        stats = self._usage_stats[provider_name]
        stats.total_requests += 1
        if response.usage:
            stats.total_prompt_tokens += response.usage.prompt_tokens or 0
            stats.total_completion_tokens += response.usage.completion_tokens or 0
            stats.total_tokens += response.usage.total_tokens or 0

    def _record_failure(self, provider_name: str) -> None:
        """Record a failed request."""
        if provider_name in self._usage_stats:
            self._usage_stats[provider_name].failed_requests += 1

    def get_usage_stats(self) -> dict[str, ProviderUsageStats]:
        """Return per-provider usage statistics."""
        return dict(self._usage_stats)

    def reset_usage_stats(self) -> None:
        """Clear all usage counters."""
        for name in self._usage_stats:
            self._usage_stats[name] = ProviderUsageStats(provider_name=name)
