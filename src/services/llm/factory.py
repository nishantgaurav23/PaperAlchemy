"""Factory functions for creating LLM providers.

create_llm_provider() picks the best available provider based on config.
create_llm_providers() returns all available providers for multi-provider access.
"""

from __future__ import annotations

import logging

import anthropic  # noqa: F401 — imported so tests can patch it
from google import genai  # noqa: F401 — imported so tests can patch it
from src.config import Settings
from src.services.llm.anthropic_provider import AnthropicProvider
from src.services.llm.gemini_provider import GeminiProvider
from src.services.llm.ollama_provider import OllamaProvider
from src.services.llm.provider import LLMProvider
from src.services.llm.router import LLMRouter

logger = logging.getLogger(__name__)


def create_llm_provider(settings: Settings) -> LLMProvider:
    """Return the preferred LLM provider based on configuration.

    Priority: Anthropic (Claude) → Gemini (Google) → Ollama (local).
    Falls back through the chain if a provider fails to initialize.
    """
    if settings.anthropic.api_key:
        try:
            provider = AnthropicProvider(settings.anthropic)
            logger.info("Using AnthropicProvider (API key configured)")
            return provider
        except Exception as exc:
            logger.warning("AnthropicProvider init failed (%s), falling back", exc)

    if settings.gemini.api_key:
        try:
            provider = GeminiProvider(settings.gemini)
            logger.info("Using GeminiProvider (API key configured)")
            return provider
        except Exception as exc:
            logger.warning("GeminiProvider init failed (%s), falling back to Ollama", exc)

    logger.info("Using OllamaProvider")
    return OllamaProvider(settings.ollama)


def create_llm_providers(settings: Settings) -> dict[str, LLMProvider]:
    """Return all available LLM providers keyed by name.

    Always includes "ollama". Includes "anthropic" and/or "gemini" if API keys are configured.
    """
    providers: dict[str, LLMProvider] = {"ollama": OllamaProvider(settings.ollama)}

    if settings.anthropic.api_key:
        providers["anthropic"] = AnthropicProvider(settings.anthropic)

    if settings.gemini.api_key:
        providers["gemini"] = GeminiProvider(settings.gemini)

    logger.info("Available LLM providers: %s", list(providers.keys()))
    return providers


def create_llm_router(settings: Settings) -> LLMRouter:
    """Create an LLMRouter with all available providers and routing config."""
    providers = create_llm_providers(settings)
    return LLMRouter(providers=providers, routing_settings=settings.llm_routing)
