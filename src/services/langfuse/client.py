"""
Langfuse V3 tracing client for PaperAlchemy pipeline observability.

Why it's needed:
    RAG pipelines have multiple stages (embed -> search -> prompt -> generate)
    and debugging slow or incorrect answers requires per-stage timing and
    metadata. Langfuse provides a hosted/self-hosted dashboard for traces.

What it does:
    - Wraps the Langfuse SDK for trae/span/generation lifecycle
    - Provides context-manager helpers for clean start/end patterns
    - Handle flush and shutodown for graceful cleanup.
    - Gracefullu degrades if Langfuse is unreachable

How it helps:
    - Full pipeline visibility in the lagfuse dashboard
    - Per-stage latency tracking identifies bottlenecks
    - Generation metadata (model, tokens, temperature) logged automatically
    - Feedback scoring enables evaluation workflows.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

class LangfuseTracer:
    """Wrapper around the Langfuse SDK for structuring tracing."""

    def __init__(self, langfuse_client: Any, enabled: bool = True):
        self._langfuse = langfuse_client
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled and self._langfuse is not None
    
    def create_trace(self, name: str, metadata: Optional[dict] = None, **kwargs) -> Optional[Any]:
        """Start a new top-level trace."""
        if not self.enabled:
            return None
        try:
            return self._langfuse.trace(name=name, metadata=metadata or {}, **kwargs)
        except Exception as e:
            logger.warning(f"Langfuse trace creation failed: {e}")
            return None
        
    def start_span(self, trace: Any, name: str, metadata: Optional[dict] = None, **kwargs) -> Optional[Any]:
        """Start a span within an existing trace."""
        if not self.enabled or trace is None:
            return None
        try:
            return trace.span(name=name, metadata=metadata or {}, **kwargs)
        except Exception as e:
            logger.warning(f"Langfuse span creation failed: {e}")
            return None
        
    def update_span(self, span: Any, output: Any = None, metadata: Optional[dict] = None, **kwargs) -> None:
        """Attach output and metadata to span."""
        if span is None:
            return
        try:
            span.end(output=output, metadata=metadata or {}, **kwargs)
        except Exception as e:
            logger.warning(f"Langfuse span update failed: {e} ")

    def start_generation(
            self,
            trace: Any,
            name: str,
            model: Optional[str] = None,
            input: Optional[Any] = None,
            metadata: Optional[dict] = None,
            **kwargs,
    ) -> Optional[Any]:
        """Start a generation span for LLM calls."""
        if not self.enabled or trace is None:
            return None
        try:
            return trace.generation(
                name=name,
                model=model,
                input=input,
                metadata=metadata or {},
                **kwargs,
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
        """Attach output and usage metrics to a generation."""
        if generation is None:
            return
        try:
            generation.end(output=output, usage=usage, metadata=metadata or {}, **kwargs)
        except Exception as e:
            logger.warning(f"Langfuse generation update failed: {e}")

    def submit_feedback(
            self,
            trace_id: str,
            name: str,
            value: float,
            comment: Optional[str] = None
    )-> None:
        """Submit user feedback for a Langfuse score."""
        if not self.enabled:
            return
        try:
            self._langfuse.score(trace_id=trace_id, name=name, value=value, comment=comment)
        except Exception as e:
            logger.warning(f"Langfuse feedback submission failed: {e}")

    def flush(self) -> None:
        """Flush pending traces to Langfuse"""
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