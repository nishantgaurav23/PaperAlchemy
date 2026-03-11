"""Tests for S7.1 — Conversation Memory.

TDD Red phase: all tests written before implementation.
Tests use mocked Redis — no real connection needed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.services.chat.memory import ChatMessage, ConversationMemory, make_conversation_memory

# ---------------------------------------------------------------------------
# ChatMessage model tests (FR-7)
# ---------------------------------------------------------------------------


class TestChatMessageModel:
    def test_serialization_roundtrip(self) -> None:
        """ChatMessage serializes to JSON and deserializes back identically."""
        msg = ChatMessage(
            role="user",
            content="What is attention?",
            timestamp=datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC),
            metadata={"source": "test"},
        )
        json_str = msg.model_dump_json()
        restored = ChatMessage.model_validate_json(json_str)
        assert restored.role == msg.role
        assert restored.content == msg.content
        assert restored.timestamp == msg.timestamp
        assert restored.metadata == msg.metadata

    def test_default_timestamp(self) -> None:
        """ChatMessage gets a default timestamp when not provided."""
        msg = ChatMessage(role="user", content="hello")
        assert msg.timestamp is not None
        assert isinstance(msg.timestamp, datetime)

    def test_default_metadata_none(self) -> None:
        """ChatMessage metadata defaults to None."""
        msg = ChatMessage(role="user", content="hello")
        assert msg.metadata is None

    def test_invalid_role_rejected(self) -> None:
        """ChatMessage rejects roles other than 'user' or 'assistant'."""
        with pytest.raises(ValueError):
            ChatMessage(role="system", content="hello")

    def test_valid_roles(self) -> None:
        """Both 'user' and 'assistant' roles are accepted."""
        user_msg = ChatMessage(role="user", content="hi")
        asst_msg = ChatMessage(role="assistant", content="hello")
        assert user_msg.role == "user"
        assert asst_msg.role == "assistant"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_redis() -> AsyncMock:
    """Return a mocked async Redis client."""
    redis = AsyncMock()
    redis.rpush = AsyncMock(return_value=1)
    redis.lrange = AsyncMock(return_value=[])
    redis.ltrim = AsyncMock(return_value=True)
    redis.llen = AsyncMock(return_value=0)
    redis.delete = AsyncMock(return_value=1)
    redis.expire = AsyncMock(return_value=True)
    redis.exists = AsyncMock(return_value=1)
    redis.scan_iter = MagicMock()  # sync method that returns async iterator
    return redis


@pytest.fixture
def memory(mock_redis: AsyncMock) -> ConversationMemory:
    """Return a ConversationMemory with mocked Redis and default settings."""
    return ConversationMemory(redis_client=mock_redis, max_messages=20, ttl_seconds=86400)


@pytest.fixture
def small_window_memory(mock_redis: AsyncMock) -> ConversationMemory:
    """ConversationMemory with a small window (3 messages) for trim tests."""
    return ConversationMemory(redis_client=mock_redis, max_messages=3, ttl_seconds=86400)


# ---------------------------------------------------------------------------
# add_message tests (FR-1)
# ---------------------------------------------------------------------------


class TestAddMessage:
    async def test_success(self, memory: ConversationMemory, mock_redis: AsyncMock) -> None:
        """add_message appends a serialized message to the Redis list."""
        await memory.add_message("sess-1", "user", "What is attention?")
        mock_redis.rpush.assert_called_once()
        call_args = mock_redis.rpush.call_args
        assert call_args[0][0] == "chat:session:sess-1"
        # Second arg is the serialized message
        stored = json.loads(call_args[0][1])
        assert stored["role"] == "user"
        assert stored["content"] == "What is attention?"

    async def test_empty_content_raises(self, memory: ConversationMemory) -> None:
        """add_message raises ValueError for empty content."""
        with pytest.raises(ValueError, match="content"):
            await memory.add_message("sess-1", "user", "")

    async def test_whitespace_content_raises(self, memory: ConversationMemory) -> None:
        """add_message raises ValueError for whitespace-only content."""
        with pytest.raises(ValueError, match="content"):
            await memory.add_message("sess-1", "user", "   ")

    async def test_invalid_role_raises(self, memory: ConversationMemory) -> None:
        """add_message raises ValueError for invalid role."""
        with pytest.raises(ValueError, match="role"):
            await memory.add_message("sess-1", "system", "hello")

    async def test_redis_error_does_not_raise(self, memory: ConversationMemory, mock_redis: AsyncMock) -> None:
        """add_message logs warning but does not raise on Redis failure."""
        mock_redis.rpush.side_effect = ConnectionError("Redis down")
        # Should not raise
        await memory.add_message("sess-1", "user", "hello")

    async def test_with_metadata(self, memory: ConversationMemory, mock_redis: AsyncMock) -> None:
        """add_message stores metadata alongside the message."""
        metadata = {"sources": [{"title": "Attention Is All You Need"}]}
        await memory.add_message("sess-1", "assistant", "Answer here", metadata=metadata)
        call_args = mock_redis.rpush.call_args
        stored = json.loads(call_args[0][1])
        assert stored["metadata"] == metadata

    async def test_refreshes_ttl(self, memory: ConversationMemory, mock_redis: AsyncMock) -> None:
        """add_message refreshes the session TTL after adding."""
        await memory.add_message("sess-1", "user", "hello")
        mock_redis.expire.assert_called_once_with("chat:session:sess-1", 86400)


# ---------------------------------------------------------------------------
# Sliding window tests (FR-2)
# ---------------------------------------------------------------------------


class TestSlidingWindow:
    async def test_trims_at_limit(self, small_window_memory: ConversationMemory, mock_redis: AsyncMock) -> None:
        """After exceeding max_messages, ltrim is called to keep only the last N."""
        mock_redis.llen.return_value = 5  # Simulate 5 messages stored, max is 3
        await small_window_memory.add_message("sess-1", "user", "msg")
        mock_redis.ltrim.assert_called_once_with("chat:session:sess-1", -3, -1)

    async def test_no_trim_under_limit(self, small_window_memory: ConversationMemory, mock_redis: AsyncMock) -> None:
        """No ltrim call when message count is within max_messages."""
        mock_redis.llen.return_value = 2  # Under the limit of 3
        await small_window_memory.add_message("sess-1", "user", "msg")
        mock_redis.ltrim.assert_not_called()

    async def test_trim_at_exact_limit(self, small_window_memory: ConversationMemory, mock_redis: AsyncMock) -> None:
        """No trimming when exactly at max_messages."""
        mock_redis.llen.return_value = 3  # Exactly at limit
        await small_window_memory.add_message("sess-1", "user", "msg")
        mock_redis.ltrim.assert_not_called()


# ---------------------------------------------------------------------------
# get_history tests (FR-3)
# ---------------------------------------------------------------------------


class TestGetHistory:
    async def test_success(self, memory: ConversationMemory, mock_redis: AsyncMock) -> None:
        """get_history returns deserialized messages in order."""
        msg1 = ChatMessage(role="user", content="Q1", timestamp=datetime(2025, 1, 1, tzinfo=UTC))
        msg2 = ChatMessage(role="assistant", content="A1", timestamp=datetime(2025, 1, 1, 0, 1, tzinfo=UTC))
        mock_redis.lrange.return_value = [msg1.model_dump_json(), msg2.model_dump_json()]

        history = await memory.get_history("sess-1")
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == "Q1"
        assert history[1].role == "assistant"
        assert history[1].content == "A1"

    async def test_empty_session(self, memory: ConversationMemory, mock_redis: AsyncMock) -> None:
        """get_history returns empty list for unknown session."""
        mock_redis.lrange.return_value = []
        history = await memory.get_history("nonexistent")
        assert history == []

    async def test_with_limit(self, memory: ConversationMemory, mock_redis: AsyncMock) -> None:
        """get_history with limit returns only the last N messages."""
        msgs = [
            ChatMessage(role="user", content=f"msg{i}", timestamp=datetime(2025, 1, 1, tzinfo=UTC)).model_dump_json()
            for i in range(5)
        ]
        mock_redis.lrange.return_value = msgs[-2:]  # Simulate Redis returning last 2

        history = await memory.get_history("sess-1", limit=2)
        assert len(history) == 2
        # Verify lrange was called with correct negative index
        mock_redis.lrange.assert_called_once_with("chat:session:sess-1", -2, -1)

    async def test_redis_error_returns_empty(self, memory: ConversationMemory, mock_redis: AsyncMock) -> None:
        """get_history returns empty list on Redis failure."""
        mock_redis.lrange.side_effect = ConnectionError("Redis down")
        history = await memory.get_history("sess-1")
        assert history == []

    async def test_refreshes_ttl(self, memory: ConversationMemory, mock_redis: AsyncMock) -> None:
        """get_history refreshes the session TTL."""
        mock_redis.lrange.return_value = []
        await memory.get_history("sess-1")
        mock_redis.expire.assert_called_once_with("chat:session:sess-1", 86400)

    async def test_corrupted_data_skipped(self, memory: ConversationMemory, mock_redis: AsyncMock) -> None:
        """Corrupted entries in Redis are skipped, valid ones returned."""
        valid = ChatMessage(role="user", content="ok", timestamp=datetime(2025, 1, 1, tzinfo=UTC))
        mock_redis.lrange.return_value = ["not-valid-json", valid.model_dump_json()]
        history = await memory.get_history("sess-1")
        assert len(history) == 1
        assert history[0].content == "ok"


# ---------------------------------------------------------------------------
# clear_session tests (FR-4)
# ---------------------------------------------------------------------------


class TestClearSession:
    async def test_success(self, memory: ConversationMemory, mock_redis: AsyncMock) -> None:
        """clear_session deletes the key and returns True."""
        mock_redis.delete.return_value = 1
        result = await memory.clear_session("sess-1")
        assert result is True
        mock_redis.delete.assert_called_once_with("chat:session:sess-1")

    async def test_not_found(self, memory: ConversationMemory, mock_redis: AsyncMock) -> None:
        """clear_session returns False for non-existent session."""
        mock_redis.delete.return_value = 0
        result = await memory.clear_session("nonexistent")
        assert result is False

    async def test_redis_error(self, memory: ConversationMemory, mock_redis: AsyncMock) -> None:
        """clear_session returns False on Redis failure."""
        mock_redis.delete.side_effect = ConnectionError("Redis down")
        result = await memory.clear_session("sess-1")
        assert result is False


# ---------------------------------------------------------------------------
# list_sessions tests (FR-6)
# ---------------------------------------------------------------------------


class TestListSessions:
    async def test_success(self, memory: ConversationMemory, mock_redis: AsyncMock) -> None:
        """list_sessions returns session IDs extracted from Redis keys."""

        async def _scan_iter(match: str):
            for key in ["chat:session:sess-1", "chat:session:sess-2"]:
                yield key

        mock_redis.scan_iter = _scan_iter
        sessions = await memory.list_sessions()
        assert sorted(sessions) == ["sess-1", "sess-2"]

    async def test_redis_error(self, memory: ConversationMemory, mock_redis: AsyncMock) -> None:
        """list_sessions returns empty list on Redis failure."""

        async def _scan_iter(match: str):
            raise ConnectionError("Redis down")
            yield  # make it an async generator  # noqa: RUF036

        mock_redis.scan_iter = _scan_iter
        sessions = await memory.list_sessions()
        assert sessions == []


# ---------------------------------------------------------------------------
# Factory tests (FR-8)
# ---------------------------------------------------------------------------


class TestFactory:
    async def test_make_conversation_memory_success(self) -> None:
        """Factory returns ConversationMemory when Redis is available."""
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)

        with patch("src.services.chat.memory.make_redis_client", new_callable=AsyncMock, return_value=mock_redis):
            result = await make_conversation_memory()
            assert isinstance(result, ConversationMemory)

    async def test_make_conversation_memory_no_redis(self) -> None:
        """Factory returns None when Redis is unavailable."""
        with patch("src.services.chat.memory.make_redis_client", new_callable=AsyncMock, return_value=None):
            result = await make_conversation_memory()
            assert result is None

    async def test_factory_uses_settings_ttl(self) -> None:
        """Factory reads TTL from Redis settings."""
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)

        with (
            patch("src.services.chat.memory.make_redis_client", new_callable=AsyncMock, return_value=mock_redis),
            patch("src.services.chat.memory.get_settings") as mock_settings,
        ):
            mock_settings.return_value.redis.ttl_hours = 12
            result = await make_conversation_memory()
            assert result is not None
            assert result._ttl == 12 * 3600
