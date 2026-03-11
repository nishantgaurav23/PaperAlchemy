"""Session-based conversation memory backed by Redis (S7.1).

Stores chat messages per session as a JSON list in Redis with a sliding
window (default 20 messages) and configurable TTL (default 24h).
All Redis failures are caught and logged — never propagated.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, model_validator
from redis.asyncio import Redis
from src.config import get_settings
from src.services.cache.factory import make_redis_client

logger = logging.getLogger(__name__)

VALID_ROLES = ("user", "assistant")
KEY_PREFIX = "chat:session:"


class ChatMessage(BaseModel):
    """A single chat message in a conversation session."""

    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime | None = None
    metadata: dict | None = None

    @model_validator(mode="after")
    def _set_default_timestamp(self) -> ChatMessage:
        if self.timestamp is None:
            self.timestamp = datetime.now(tz=UTC)
        return self


class ConversationMemory:
    """Redis-backed sliding-window conversation history."""

    def __init__(self, redis_client: Redis, max_messages: int = 20, ttl_seconds: int = 86400) -> None:
        self._redis = redis_client
        self._max_messages = max_messages
        self._ttl = ttl_seconds

    @staticmethod
    def _key(session_id: str) -> str:
        return f"{KEY_PREFIX}{session_id}"

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> None:
        """Append a message to the session history. Never raises on Redis failure."""
        if not content or not content.strip():
            msg = "content must not be empty"
            raise ValueError(msg)
        if role not in VALID_ROLES:
            msg = f"role must be one of {VALID_ROLES}, got '{role}'"
            raise ValueError(msg)

        message = ChatMessage(role=role, content=content, metadata=metadata)  # type: ignore[arg-type]
        key = self._key(session_id)

        try:
            await self._redis.rpush(key, message.model_dump_json())

            # Sliding window: trim if over limit
            length = await self._redis.llen(key)
            if length > self._max_messages:
                await self._redis.ltrim(key, -self._max_messages, -1)

            # Refresh TTL
            await self._redis.expire(key, self._ttl)
            logger.debug("Added %s message to session %s", role, session_id)
        except Exception:
            logger.warning("Failed to add message to session %s (graceful skip)", session_id, exc_info=True)

    async def get_history(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[ChatMessage]:
        """Return conversation history for a session. Returns [] on error."""
        key = self._key(session_id)
        try:
            if limit is not None:
                raw_messages = await self._redis.lrange(key, -limit, -1)
            else:
                raw_messages = await self._redis.lrange(key, 0, -1)

            messages: list[ChatMessage] = []
            for raw in raw_messages:
                try:
                    messages.append(ChatMessage.model_validate_json(raw))
                except Exception:
                    logger.warning("Skipping corrupted message in session %s", session_id)

            # Refresh TTL
            await self._redis.expire(key, self._ttl)
            return messages
        except Exception:
            logger.warning("Failed to get history for session %s (graceful skip)", session_id, exc_info=True)
            return []

    async def clear_session(self, session_id: str) -> bool:
        """Delete all messages for a session. Returns True if session existed."""
        try:
            deleted = await self._redis.delete(self._key(session_id))
            return deleted > 0
        except Exception:
            logger.warning("Failed to clear session %s (graceful skip)", session_id, exc_info=True)
            return False

    async def list_sessions(self) -> list[str]:
        """Return all active session IDs. Returns [] on error."""
        try:
            sessions: list[str] = []
            async for key in self._redis.scan_iter(match=f"{KEY_PREFIX}*"):
                # Extract session_id from "chat:session:{session_id}"
                session_id = key.removeprefix(KEY_PREFIX) if isinstance(key, str) else key.decode().removeprefix(KEY_PREFIX)
                sessions.append(session_id)
            return sessions
        except Exception:
            logger.warning("Failed to list sessions (graceful skip)", exc_info=True)
            return []


async def make_conversation_memory() -> ConversationMemory | None:
    """Create a ConversationMemory backed by Redis.

    Returns None if Redis is unavailable (graceful degradation).
    """
    redis_client = await make_redis_client()
    if redis_client is None:
        return None

    settings = get_settings()
    ttl_seconds = settings.redis.ttl_hours * 3600
    return ConversationMemory(redis_client=redis_client, max_messages=20, ttl_seconds=ttl_seconds)
