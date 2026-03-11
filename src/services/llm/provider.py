"""LLM provider protocol and shared response models.

Defines the interface that OllamaProvider and GeminiProvider both implement,
plus Pydantic models for responses, usage metadata, and health checks.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class UsageMetadata(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float | None = None


class LLMResponse(BaseModel):
    text: str
    model: str
    provider: str
    usage: UsageMetadata | None = None


class HealthStatus(BaseModel):
    healthy: bool
    provider: str
    message: str
    version: str | None = None


@runtime_checkable
class LLMProvider(Protocol):
    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...

    async def generate_stream(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]: ...

    async def health_check(self) -> HealthStatus: ...

    def get_langchain_model(
        self,
        *,
        model: str | None = None,
        temperature: float | None = None,
    ): ...

    async def close(self) -> None: ...
