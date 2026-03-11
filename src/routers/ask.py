"""SSE streaming + non-streaming /ask endpoint (S5.3).

Exposes RAGChain.aquery() and RAGChain.aquery_stream() as a single
POST /ask endpoint. Streaming mode (default) uses Server-Sent Events;
non-streaming mode returns a JSON AskResponse.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.dependency import RAGChainDep
from src.exceptions import LLMServiceError, PaperAlchemyError
from src.schemas.api.ask import AskRequest, AskResponse

logger = logging.getLogger(__name__)

router = APIRouter()

_SOURCES_MARKER = "\n\n[SOURCES]"


def _sse_event(event: str, data: str) -> str:
    """Format a single SSE event."""
    return f"event: {event}\ndata: {data}\n\n"


async def _stream_rag(rag_chain, request: AskRequest) -> AsyncGenerator[str]:
    """Yield SSE events from the RAG chain stream."""
    sources_json: str | None = None
    try:
        async for chunk in rag_chain.aquery_stream(
            request.query,
            top_k=request.top_k,
            categories=request.categories,
            temperature=request.temperature,
        ):
            # Check if this chunk contains the [SOURCES] marker
            if _SOURCES_MARKER in chunk:
                parts = chunk.split(_SOURCES_MARKER, 1)
                text_part = parts[0]
                sources_json = parts[1]
                if text_part:
                    yield _sse_event("token", json.dumps({"text": text_part}))
            else:
                yield _sse_event("token", json.dumps({"text": chunk}))

        # Emit sources event
        if sources_json is not None:
            yield _sse_event("sources", sources_json)
        else:
            yield _sse_event("sources", "[]")

        yield _sse_event("done", "{}")

    except (LLMServiceError, PaperAlchemyError) as exc:
        logger.warning("RAG stream error: %s", exc)
        yield _sse_event("error", json.dumps({"error": str(exc)}))
    except Exception:
        logger.exception("Unexpected error during RAG stream")
        yield _sse_event("error", json.dumps({"error": "Internal server error"}))


@router.post("/ask")
async def ask(request: AskRequest, rag_chain: RAGChainDep):
    """Ask a research question. Streams SSE by default; set stream=false for JSON."""
    if request.stream:
        return StreamingResponse(
            _stream_rag(rag_chain, request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming: call aquery and return JSON
    try:
        rag_response = await rag_chain.aquery(
            request.query,
            top_k=request.top_k,
            categories=request.categories,
            temperature=request.temperature,
        )
    except LLMServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return AskResponse(
        answer=rag_response.answer,
        sources=rag_response.sources,
        query=rag_response.query,
        retrieval_metadata=rag_response.retrieval_metadata,
        llm_metadata=rag_response.llm_metadata,
    )
