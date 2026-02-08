"""
Factory function for creating OllamaClient instances.

Why it's needed:
    The OllamaClient requires Settings for configuration (host, port, model 
    defaults). The factory centralizes this initialization so the lifespan
    manager and tests both create clients the same way. 

What it does:
    - make_ollama_client(): Creates an OllamaClient with settings from
      environment variables or defaults.

How it helps:
    - Single place to change if we swap LLM providers
    - Consistent initialization across lifespan and tests
    - Testing: can pbe patched to return a mock client
"""

from typing import Optional

from src.config import Settings, get_settings
from src.services.ollama.client import OllamaClient

def make_ollama_client(settings: Optional[Settings] = None) -> OllamaClient:
    """Create a new OllamaClient instance.
    
    Args:
        settings: Optional Settings instance. If None, loads from
                   environment variables via get_settings().
    
    Returns:
        OllamaClient configured with host, port, and generation defaults.
    """

    if settings is None:
        settings = get_settings()

    return OllamaClient(settings=settings)