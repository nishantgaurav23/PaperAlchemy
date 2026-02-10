"""
Factory function for creating LangFuseTracer instances.

Why it's needed:
    Centralizes the Langfuse SDK initialization with settings from enviornment.
    The lifespan manager calls this once at startup. Returns a disabled
    tracer if Langfuse it not configured or unreachable.

Why it does:
    - make_langfuse_tracer(): Creates a LangfuseTracer backed by the 
      Langfuse SDK client, or a disabled stub if not configured.

How it helps:
    - Single place for Langfuse connection configuration
    - Graceful degradation: missing keys -> disabled tracer, not crash
    - RAGTracer wraps the base tracer for pipeline-specific methods.
"""

import logging
from typing import Optional

from src.config import Settings, get_settings
from src.services.langfuse.client import LangfuseTracer

logger = logging.getLogger(__name__)

def make_langfuse_tracer(settings: Optional[Settings] = None) -> LangfuseTracer:
    """Create a LangfuseTracer instances.
    
    Returns a disabled tracer if langfuse is not enabled or keys are missing.
    """

    if settings is None:
        settings = get_settings()

    ls = settings.langfuse

    if not ls.enabled or not ls.public_key or not ls.secret_key:
        logger.info("Langfuse disabled (enabled=False or missing keys)")
        return LangfuseTracer(langfuse_client=None, enabled=False)
    
    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_keys=ls.public_key,
            secret_keys=ls.secret_key,
            host=ls.host,
            flush_at=ls.flush_at,
            flush_interval=ls.flush_interval,
            debug=ls.debug,
            max_retries=ls.max_retries,
            timeout=ls.timeout,
        )
        logger.info(f"Langfuse connected at {ls.host}")
        return LangfuseTracer(langfuse_client=client, enabled=True)
    except Exception as e:
        logger.warning(f"Langfuse initialization failed: (tracing disabled): {e}")
        return LangfuseTracer(langfuse_client=None, enabled=False)
