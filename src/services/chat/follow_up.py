"""Context-aware follow-up Q&A handler (S7.2).

Detects follow-up queries via heuristics, resolves coreferences using the LLM,
and re-retrieves from the knowledge base every time. Conversation history is
stored in Redis via ConversationMemory (S7.1).
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator

from pydantic import BaseModel
from src.services.chat.memory import ChatMessage, ConversationMemory
from src.services.llm.provider import LLMProvider
from src.services.rag.chain import RAGChain
from src.services.rag.models import RAGResponse

logger = logging.getLogger(__name__)

# Pronouns that suggest coreference with prior context
_COREFERENCE_PRONOUNS = re.compile(
    r"\b(it|its|they|them|their|this|that|those|these)\b",
    re.IGNORECASE,
)

# Continuation phrases at the start of a query
_CONTINUATION_PREFIXES = (
    "what about",
    "how about",
    "and ",
    "also ",
    "but ",
    "can you",
)

_SHORT_QUERY_THRESHOLD = 5  # words

_REWRITE_PROMPT_TEMPLATE = """\
You are a query rewriter. Given a conversation history and a follow-up question, \
rewrite the follow-up into a fully self-contained question that resolves all pronouns \
and references using the conversation context.

Return ONLY the rewritten question — no explanation, no preamble.

Conversation history:
{history}

Follow-up question: {query}

Rewritten question:"""


class FollowUpResult(BaseModel):
    """Result of follow-up handling: tracks original/rewritten query and response."""

    original_query: str
    rewritten_query: str
    is_follow_up: bool
    response: RAGResponse | None = None


def is_follow_up(query: str, history: list[ChatMessage]) -> bool:
    """Heuristic follow-up detection — fast, no LLM call.

    Returns True if the query likely references prior conversation context.
    Always returns False when history is empty.
    """
    if not history:
        return False

    query_lower = query.lower().strip()

    # Check continuation prefixes
    for prefix in _CONTINUATION_PREFIXES:
        if query_lower.startswith(prefix):
            return True

    # Check coreference pronouns
    if _COREFERENCE_PRONOUNS.search(query):
        return True

    # Very short queries with history present
    return len(query.split()) < _SHORT_QUERY_THRESHOLD


async def rewrite_query(
    query: str,
    history: list[ChatMessage],
    llm_provider: LLMProvider,
    *,
    max_history_messages: int = 10,
) -> str:
    """Rewrite a follow-up query into a self-contained question using LLM.

    Falls back to original query on empty history, LLM failure, or empty response.
    Only uses the last ``max_history_messages`` from history.
    """
    if not history:
        return query

    # Trim history to last N messages
    trimmed = history[-max_history_messages:]

    # Format history for the prompt
    history_lines: list[str] = []
    for msg in trimmed:
        role_label = "User" if msg.role == "user" else "Assistant"
        history_lines.append(f"{role_label}: {msg.content}")

    prompt = _REWRITE_PROMPT_TEMPLATE.format(
        history="\n".join(history_lines),
        query=query,
    )

    try:
        response = await llm_provider.generate(prompt, temperature=0.0, max_tokens=256)
        rewritten = response.text.strip()
        if not rewritten:
            logger.warning("LLM returned empty rewrite, falling back to original query")
            return query
        return rewritten
    except Exception:
        logger.warning("LLM rewrite failed, falling back to original query", exc_info=True)
        return query


class FollowUpHandler:
    """Orchestrates follow-up Q&A: detect → rewrite → RAG → store."""

    def __init__(
        self,
        *,
        llm_provider: LLMProvider,
        rag_chain: RAGChain,
        memory: ConversationMemory | None = None,
    ) -> None:
        self._llm = llm_provider
        self._rag = rag_chain
        self._memory = memory

    async def _get_history(self, session_id: str) -> list[ChatMessage]:
        if self._memory is None:
            return []
        return await self._memory.get_history(session_id)

    async def _store_exchange(self, session_id: str, user_query: str, assistant_answer: str) -> None:
        if self._memory is None:
            return
        await self._memory.add_message(session_id=session_id, role="user", content=user_query)
        await self._memory.add_message(session_id=session_id, role="assistant", content=assistant_answer)

    async def _resolve_query(self, query: str, history: list[ChatMessage]) -> tuple[str, bool]:
        """Detect follow-up and rewrite if needed. Returns (resolved_query, is_follow_up)."""
        follow_up = is_follow_up(query, history)
        if follow_up:
            rewritten = await rewrite_query(query, history, self._llm)
            return rewritten, True
        return query, False

    async def handle(
        self,
        session_id: str,
        query: str,
        *,
        top_k: int | None = None,
        categories: list[str] | None = None,
        temperature: float | None = None,
    ) -> FollowUpResult:
        """Full follow-up pipeline: detect → rewrite → RAG → store."""
        history = await self._get_history(session_id)
        resolved_query, follow_up = await self._resolve_query(query, history)

        response = await self._rag.aquery(
            resolved_query,
            top_k=top_k,
            categories=categories,
            temperature=temperature,
        )

        await self._store_exchange(session_id, query, response.answer)

        return FollowUpResult(
            original_query=query,
            rewritten_query=resolved_query,
            is_follow_up=follow_up,
            response=response,
        )

    async def handle_stream(
        self,
        session_id: str,
        query: str,
        *,
        top_k: int | None = None,
        categories: list[str] | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Streaming follow-up pipeline. Stores messages after stream is consumed."""
        history = await self._get_history(session_id)
        resolved_query, _ = await self._resolve_query(query, history)

        raw_stream = self._rag.aquery_stream(
            resolved_query,
            top_k=top_k,
            categories=categories,
            temperature=temperature,
        )

        # aquery_stream is an async generator — if mocked with AsyncMock it
        # returns a coroutine; in production it returns an async iterator.
        # Normalise to async iterator.
        if hasattr(raw_stream, "__aiter__"):
            stream_iter = raw_stream
        else:
            stream_iter = await raw_stream  # type: ignore[misc]

        async def _wrapper() -> AsyncIterator[str]:
            collected: list[str] = []
            async for token in stream_iter:
                collected.append(token)
                yield token

            # Store exchange after stream is fully consumed
            full_answer = "".join(collected)
            await self._store_exchange(session_id, query, full_answer)

        return _wrapper()
