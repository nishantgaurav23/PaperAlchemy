"""
Langfuse V3 tracing client for PaperAlchemy pipeline observability.

Why it's needed:
    RAG pipelines have multiple stages (embed -> search -> prompt -> generate)
    and debugging slow or incorrect answers requires per-stage timing and
    metadata. Langfuse provides a hosted/self-hosted dashboard for traces.

What it does:
    - Wraps the Langfuse v3 SDK for span/generation lifecycle
    - Provides helpers for clean start/end patterns
    - Handles flush and shutdown for graceful cleanup
    - Gracefully degrades if Langfuse is unreachable

v3 API change summary (from v2):
    v2: langfuse.trace() → trace.span() → trace.generation()
    v3: langfuse.start_span() → span.start_span() → span.start_generation()
    Both return objects with .update() and .end() methods.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LangfuseTracer:
    """Wrapper around the Langfuse v3 SDK for structuring pipeline tracing."""

    def __init__(self, langfuse_client: Any, enabled: bool = True):
        self._langfuse = langfuse_client
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled and self._langfuse is not None

    def create_trace(self, name: str, metadata: Optional[dict] = None, **kwargs) -> Optional[Any]:
        """Start a new root span (acts as a trace in v3)."""
        if not self.enabled:
            return None
        try:
            return self._langfuse.start_span(
                name=name,
                metadata=metadata or {},
                **{k: v for k, v in kwargs.items() if k in ("input", "output", "version", "level")},
            )
        except Exception as e:
            logger.warning(f"Langfuse trace creation failed: {e}")
            return None

    def start_span(self, trace: Any, name: str, metadata: Optional[dict] = None, **kwargs) -> Optional[Any]:
        """Start a child span within an existing trace/span."""
        if not self.enabled or trace is None:
            return None
        try:
            return trace.start_span(
                name=name,
                metadata=metadata or {},
                **{k: v for k, v in kwargs.items() if k in ("input", "output", "version", "level")},
            )
        except Exception as e:
            logger.warning(f"Langfuse span creation failed: {e}")
            return None

    def update_span(self, span: Any, output: Any = None, metadata: Optional[dict] = None, **kwargs) -> None:
        """Attach output and metadata to a span, then end it."""
        if span is None:
            return
        try:
            span.update(output=output, metadata=metadata or {})
            span.end()
        except Exception as e:
            logger.warning(f"Langfuse span update failed: {e}")

    def start_generation(
            self,
            trace: Any,
            name: str,
            model: Optional[str] = None,
            input: Optional[Any] = None,
            metadata: Optional[dict] = None,
            **kwargs,
    ) -> Optional[Any]:
        """Start a generation child span for LLM calls."""
        if not self.enabled or trace is None:
            return None
        try:
            return trace.start_generation(
                name=name,
                model=model,
                input=input,
                metadata=metadata or {},
            )
        except Exception as e:
            logger.warning(f"Langfuse generation creation failed: {e}")
            return None

    def update_generation(
            self,
            generation: Any,
            output: Any = None,
            usage: Optional[dict] = None,
            metadata: Optional[dict] = None,
            **kwargs,
    ) -> None:
        """Attach output and usage metrics to a generation, then end it."""
        if generation is None:
            return
        try:
            generation.update(output=output, metadata=metadata or {})
            generation.end()
        except Exception as e:
            logger.warning(f"Langfuse generation update failed: {e}")

    def submit_feedback(
            self,
            trace_id: str,
            name: str,
            value: float,
            comment: Optional[str] = None,
    ) -> None:
        """Submit a score for a trace."""
        if not self.enabled:
            return
        try:
            self._langfuse.create_score(
                trace_id=trace_id,
                name=name,
                value=value,
                comment=comment,
            )
        except Exception as e:
            logger.warning(f"Langfuse feedback submission failed: {e}")

    def flush(self) -> None:
        """Flush pending events to Langfuse."""
        if not self.enabled:
            return
        try:
            self._langfuse.flush()
        except Exception as e:
            logger.warning(f"Langfuse flush failed: {e}")

    def shutdown(self) -> None:
        """Flush and shutdown the Langfuse client."""
        if not self.enabled:
            return
        try:
            self._langfuse.flush()
            self._langfuse.shutdown()
            logger.info("Langfuse client shut down")
        except Exception as e:
            logger.warning(f"Langfuse shutdown failed: {e}")
