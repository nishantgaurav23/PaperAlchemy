"""Anthropic Claude LLM provider — wraps the Anthropic Python SDK.

Uses anthropic.AsyncAnthropic for generation and streaming, and
langchain_anthropic.ChatAnthropic for LangGraph agent nodes.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

import anthropic
from langchain_anthropic import ChatAnthropic
from src.config import AnthropicSettings
from src.exceptions import ConfigurationError, LLMConnectionError, LLMServiceError, LLMTimeoutError
from src.services.llm.provider import HealthStatus, LLMResponse, UsageMetadata

logger = logging.getLogger(__name__)


class AnthropicProvider:
    """LLMProvider implementation for Anthropic Claude cloud inference."""

    def __init__(self, settings: AnthropicSettings) -> None:
        if not settings.api_key:
            raise ConfigurationError("ANTHROPIC__API_KEY is required for AnthropicProvider")

        self._settings = settings
        self._default_model = settings.model
        self._default_temperature = settings.temperature
        self._max_tokens = settings.max_tokens
        self._client = anthropic.AsyncAnthropic(
            api_key=settings.api_key,
            timeout=float(settings.timeout),
        )

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        model_name = model or self._default_model

        try:
            message = await self._client.messages.create(
                model=model_name,
                max_tokens=max_tokens or self._max_tokens,
                temperature=temperature if temperature is not None else self._default_temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(block.text for block in message.content if block.type == "text")
            usage = self._extract_usage(message)
            return LLMResponse(text=text, model=message.model, provider="anthropic", usage=usage)
        except anthropic.AuthenticationError as exc:
            raise ConfigurationError(f"Anthropic authentication failed: {exc}") from exc
        except anthropic.APITimeoutError as exc:
            raise LLMTimeoutError(f"Anthropic generation timed out: {exc}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMConnectionError(f"Cannot connect to Anthropic API: {exc}") from exc
        except (LLMServiceError, ConfigurationError):
            raise
        except Exception as exc:
            raise LLMServiceError(f"Anthropic generation failed: {exc}") from exc

    async def generate_stream(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        model_name = model or self._default_model

        try:
            async with self._client.messages.stream(
                model=model_name,
                max_tokens=max_tokens or self._max_tokens,
                temperature=temperature if temperature is not None else self._default_temperature,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except anthropic.AuthenticationError as exc:
            raise ConfigurationError(f"Anthropic authentication failed: {exc}") from exc
        except anthropic.APITimeoutError as exc:
            raise LLMTimeoutError(f"Anthropic streaming timed out: {exc}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMConnectionError(f"Cannot connect to Anthropic API: {exc}") from exc
        except (LLMServiceError, ConfigurationError):
            raise
        except Exception as exc:
            raise LLMServiceError(f"Anthropic streaming failed: {exc}") from exc

    async def health_check(self) -> HealthStatus:
        try:
            page = await self._client.models.list()
            model_count = len(page.data)
            return HealthStatus(
                healthy=True,
                provider="anthropic",
                message=f"Anthropic API accessible ({model_count} models available)",
            )
        except Exception as exc:
            return HealthStatus(healthy=False, provider="anthropic", message=f"Anthropic API error: {exc}")

    def get_langchain_model(self, *, model: str | None = None, temperature: float | None = None) -> ChatAnthropic:
        return ChatAnthropic(
            model=model or self._default_model,
            anthropic_api_key=self._settings.api_key,
            temperature=temperature if temperature is not None else self._default_temperature,
            max_tokens=self._max_tokens,
        )

    async def close(self) -> None:
        await self._client.close()

    @staticmethod
    def _extract_usage(message) -> UsageMetadata | None:
        usage = getattr(message, "usage", None)
        if usage is None:
            return None
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        total = (input_tokens or 0) + (output_tokens or 0)
        return UsageMetadata(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=total,
        )
