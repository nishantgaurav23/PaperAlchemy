"""Factory functions for creating LLM providers.

create_llm_provider() picks the best available provider based on config.
create_llm_providers() returns all available providers for multi-provider access.
"""

from __future__ import annotations

import logging

from google import genai  # noqa: F401 — imported so tests can patch it
from src.config import Settings
from src.services.llm.gemini_provider import GeminiProvider
from src.services.llm.ollama_provider import OllamaProvider
from src.services.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


def create_llm_provider(settings: Settings) -> LLMProvider:
    """Return the preferred LLM provider based on configuration.

    If GEMINI__API_KEY is set, returns GeminiProvider (cloud).
    Otherwise, returns OllamaProvider (local).
    """
    if settings.gemini.api_key:
        logger.info("Using GeminiProvider (API key configured)")
        return GeminiProvider(settings.gemini)

    logger.info("Using OllamaProvider (no Gemini API key)")
    return OllamaProvider(settings.ollama)


def create_llm_providers(settings: Settings) -> dict[str, LLMProvider]:
    """Return all available LLM providers keyed by name.

    Always includes "ollama". Includes "gemini" only if API key is configured.
    """
    providers: dict[str, LLMProvider] = {"ollama": OllamaProvider(settings.ollama)}

    if settings.gemini.api_key:
        providers["gemini"] = GeminiProvider(settings.gemini)

    logger.info("Available LLM providers: %s", list(providers.keys()))
    return providers
