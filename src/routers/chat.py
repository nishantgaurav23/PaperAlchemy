"""Chat endpoint with session management (S7.3).

POST /chat — conversational research Q&A with follow-up detection,
coreference resolution, and citation-backed responses.
Supports both SSE streaming and JSON modes.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.dependency import ConversationMemoryDep, FollowUpHandlerDep
from src.schemas.api.chat import (
    ChatMessageOut,
    ChatRequest,
    ChatResponse,
    SessionClearResponse,
    SessionHistoryResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_SOURCES_MARKER = "\n\n[SOURCES]"


def _sse_event(event: str, data: str) -> str:
    """Format a single SSE event."""
    return f"event: {event}\ndata: {data}\n\n"


async def _stream_chat(
    handler,
    request: ChatRequest,
) -> AsyncGenerator[str]:
    """Yield SSE events from the follow-up handler stream."""
    try:
        # Emit metadata event (lightweight — don't pre-fetch history to avoid double load)
        metadata = {
            "session_id": request.session_id,
            "is_follow_up": False,
            "rewritten_query": None,
        }
        yield _sse_event("metadata", json.dumps(metadata))

        # Stream tokens from follow-up handler
        stream_iter = await handler.handle_stream(
            request.session_id,
            request.query,
            top_k=request.top_k,
            categories=request.categories,
            temperature=request.temperature,
        )

        sources_json: str | None = None
        async for chunk in stream_iter:
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

    except Exception as exc:
        logger.exception("Chat stream error: %s", exc)
        yield _sse_event("error", json.dumps({"detail": str(exc)}))


@router.post("/chat")
async def chat(request: ChatRequest, handler: FollowUpHandlerDep):
    """Chat with the research assistant. Streams SSE by default; set stream=false for JSON."""
    if request.stream:
        return StreamingResponse(
            _stream_chat(handler, request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming: call handle and return JSON
    try:
        result = await handler.handle(
            request.session_id,
            request.query,
            top_k=request.top_k,
            categories=request.categories,
            temperature=request.temperature,
        )
    except Exception as exc:
        logger.exception("Chat error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    response = result.response
    return ChatResponse(
        answer=response.answer if response else "",
        sources=response.sources if response else [],
        session_id=request.session_id,
        is_follow_up=result.is_follow_up,
        rewritten_query=result.rewritten_query if result.is_follow_up else None,
        query=result.original_query,
    )


@router.get("/chat/sessions/{session_id}/history")
async def get_session_history(session_id: str, memory: ConversationMemoryDep):
    """Retrieve conversation history for a session."""
    if memory is None:
        return SessionHistoryResponse(session_id=session_id, messages=[])

    messages = await memory.get_history(session_id)
    return SessionHistoryResponse(
        session_id=session_id,
        messages=[
            ChatMessageOut(
                role=msg.role,
                content=msg.content,
                timestamp=msg.timestamp.isoformat() if msg.timestamp else None,
            )
            for msg in messages
        ],
    )


@router.delete("/chat/sessions/{session_id}")
async def clear_session(session_id: str, memory: ConversationMemoryDep):
    """Clear all messages in a session. Idempotent — always returns 200."""
    if memory is None:
        return SessionClearResponse(session_id=session_id, cleared=False)

    cleared = await memory.clear_session(session_id)
    return SessionClearResponse(session_id=session_id, cleared=cleared)
