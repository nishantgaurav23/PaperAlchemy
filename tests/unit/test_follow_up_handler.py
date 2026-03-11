"""Tests for S7.2 — Follow-up Handler.

TDD: All tests written before implementation.
Covers: coreference detection, query rewriting, orchestration, streaming.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from src.services.chat.memory import ChatMessage
from src.services.rag.models import RAGResponse, SourceReference

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_history(pairs: list[tuple[str, str]]) -> list[ChatMessage]:
    """Build a conversation history from (user, assistant) pairs."""
    messages: list[ChatMessage] = []
    for user_msg, assistant_msg in pairs:
        messages.append(ChatMessage(role="user", content=user_msg))
        messages.append(ChatMessage(role="assistant", content=assistant_msg))
    return messages


def _make_rag_response(answer: str = "Test answer [1].", query: str = "test") -> RAGResponse:
    return RAGResponse(
        answer=answer,
        sources=[
            SourceReference(
                index=1,
                arxiv_id="1706.03762",
                title="Attention Is All You Need",
                authors=["Vaswani"],
                arxiv_url="https://arxiv.org/abs/1706.03762",
            )
        ],
        query=query,
    )


async def _mock_stream(*tokens: str) -> AsyncIterator[str]:
    for t in tokens:
        yield t


# ===========================================================================
# FR-2: Follow-up Detection (is_follow_up)
# ===========================================================================


class TestIsFollowUp:
    """Heuristic follow-up detection — no LLM call."""

    def test_with_pronoun_its(self):
        from src.services.chat.follow_up import is_follow_up

        history = _make_history([("Explain transformers", "Transformers use self-attention...")])
        assert is_follow_up("What about its limitations?", history) is True

    def test_with_pronoun_them(self):
        from src.services.chat.follow_up import is_follow_up

        history = _make_history([("Explain GANs", "GANs are generative models...")])
        assert is_follow_up("How do you train them?", history) is True

    def test_with_pronoun_this(self):
        from src.services.chat.follow_up import is_follow_up

        history = _make_history([("What is BERT?", "BERT is a language model...")])
        assert is_follow_up("Is this better than GPT?", history) is True

    def test_with_continuation_what_about(self):
        from src.services.chat.follow_up import is_follow_up

        history = _make_history([("Explain CNNs", "CNNs use convolutional filters...")])
        assert is_follow_up("What about compared to BERT?", history) is True

    def test_with_continuation_how_about(self):
        from src.services.chat.follow_up import is_follow_up

        history = _make_history([("Explain RNNs", "RNNs process sequences...")])
        assert is_follow_up("How about LSTMs?", history) is True

    def test_with_continuation_also(self):
        from src.services.chat.follow_up import is_follow_up

        history = _make_history([("Explain attention", "Attention mechanisms...")])
        assert is_follow_up("Also explain multi-head attention", history) is True

    def test_short_query_with_history(self):
        from src.services.chat.follow_up import is_follow_up

        history = _make_history([("What is BERT?", "BERT is...")])
        assert is_follow_up("Why?", history) is True

    def test_standalone_no_follow_up(self):
        from src.services.chat.follow_up import is_follow_up

        history = _make_history([("Explain CNNs", "CNNs use convolutional filters...")])
        assert is_follow_up("Explain transformer architecture in detail", history) is False

    def test_no_history_always_false(self):
        from src.services.chat.follow_up import is_follow_up

        assert is_follow_up("What about its limitations?", []) is False

    def test_no_history_short_query_false(self):
        from src.services.chat.follow_up import is_follow_up

        assert is_follow_up("Why?", []) is False

    def test_with_continuation_but(self):
        from src.services.chat.follow_up import is_follow_up

        history = _make_history([("What is GPT?", "GPT is a language model...")])
        assert is_follow_up("But how does it compare to BERT?", history) is True

    def test_with_continuation_can_you(self):
        from src.services.chat.follow_up import is_follow_up

        history = _make_history([("Explain attention", "Attention is...")])
        assert is_follow_up("Can you elaborate on that?", history) is True


# ===========================================================================
# FR-1: Query Rewriting
# ===========================================================================


class TestRewriteQuery:
    """LLM-based coreference resolution."""

    @pytest.mark.asyncio
    async def test_resolves_coreference(self):
        from src.services.chat.follow_up import rewrite_query

        mock_llm = AsyncMock()
        mock_llm.generate.return_value = MagicMock(text="What are the limitations of the Transformer architecture?")

        history = _make_history([("Explain transformers", "Transformers use self-attention...")])
        result = await rewrite_query("What about its limitations?", history, mock_llm)

        assert result == "What are the limitations of the Transformer architecture?"
        mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_standalone_returned_unchanged(self):
        from src.services.chat.follow_up import rewrite_query

        mock_llm = AsyncMock()
        mock_llm.generate.return_value = MagicMock(text="Explain transformer architecture in detail")

        history = _make_history([("What is CNN?", "CNN is...")])
        result = await rewrite_query("Explain transformer architecture in detail", history, mock_llm)

        assert result == "Explain transformer architecture in detail"

    @pytest.mark.asyncio
    async def test_empty_history_returns_original(self):
        from src.services.chat.follow_up import rewrite_query

        mock_llm = AsyncMock()
        result = await rewrite_query("What about its limitations?", [], mock_llm)

        assert result == "What about its limitations?"
        mock_llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_failure_returns_original(self):
        from src.services.chat.follow_up import rewrite_query

        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = RuntimeError("LLM down")

        history = _make_history([("Explain transformers", "Transformers...")])
        result = await rewrite_query("What about its limitations?", history, mock_llm)

        assert result == "What about its limitations?"

    @pytest.mark.asyncio
    async def test_trims_history_to_max_messages(self):
        from src.services.chat.follow_up import rewrite_query

        mock_llm = AsyncMock()
        mock_llm.generate.return_value = MagicMock(text="Rewritten query")

        # Build 20 messages (10 pairs), set max_history=4 (last 4 messages = 2 pairs)
        history = _make_history([(f"Q{i}", f"A{i}") for i in range(10)])
        await rewrite_query("What about it?", history, mock_llm, max_history_messages=4)

        call_args = mock_llm.generate.call_args
        prompt = call_args[0][0] if call_args[0] else call_args[1].get("prompt", "")
        # Should NOT contain early messages
        assert "Q0" not in prompt
        assert "Q1" not in prompt
        # Should contain recent messages
        assert "Q9" in prompt
        assert "A9" in prompt

    @pytest.mark.asyncio
    async def test_llm_returns_empty_falls_back(self):
        from src.services.chat.follow_up import rewrite_query

        mock_llm = AsyncMock()
        mock_llm.generate.return_value = MagicMock(text="   ")

        history = _make_history([("Explain transformers", "Transformers...")])
        result = await rewrite_query("What about its limitations?", history, mock_llm)

        assert result == "What about its limitations?"


# ===========================================================================
# FollowUpResult model
# ===========================================================================


class TestFollowUpResult:
    def test_fields(self):
        from src.services.chat.follow_up import FollowUpResult

        result = FollowUpResult(
            original_query="its limitations?",
            rewritten_query="What are the limitations of transformers?",
            is_follow_up=True,
            response=_make_rag_response(),
        )
        assert result.original_query == "its limitations?"
        assert result.rewritten_query == "What are the limitations of transformers?"
        assert result.is_follow_up is True
        assert result.response is not None

    def test_standalone_query_result(self):
        from src.services.chat.follow_up import FollowUpResult

        result = FollowUpResult(
            original_query="Explain transformers",
            rewritten_query="Explain transformers",
            is_follow_up=False,
            response=_make_rag_response(),
        )
        assert result.original_query == result.rewritten_query
        assert result.is_follow_up is False


# ===========================================================================
# FR-3: Follow-up Orchestration (FollowUpHandler)
# ===========================================================================


class TestFollowUpHandler:
    def _make_handler(
        self,
        *,
        llm: AsyncMock | None = None,
        rag: AsyncMock | None = None,
        memory: AsyncMock | None = None,
    ):
        from src.services.chat.follow_up import FollowUpHandler

        llm = llm or AsyncMock()
        rag = rag or AsyncMock()
        return FollowUpHandler(llm_provider=llm, rag_chain=rag, memory=memory)

    @pytest.mark.asyncio
    async def test_handle_standalone_query(self):
        """Standalone query goes through RAG without rewrite."""
        rag_mock = AsyncMock()
        rag_mock.aquery.return_value = _make_rag_response()

        handler = self._make_handler(rag=rag_mock)
        result = await handler.handle(session_id="s1", query="Explain transformer architecture in detail")

        assert result.is_follow_up is False
        assert result.original_query == "Explain transformer architecture in detail"
        assert result.rewritten_query == "Explain transformer architecture in detail"
        assert result.response is not None
        rag_mock.aquery.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_follow_up_query(self):
        """Follow-up query gets rewritten then sent to RAG."""
        memory_mock = AsyncMock()
        memory_mock.get_history.return_value = _make_history([("Explain transformers", "Transformers use self-attention [1].")])

        llm_mock = AsyncMock()
        llm_mock.generate.return_value = MagicMock(text="What are the limitations of the Transformer architecture?")

        rag_mock = AsyncMock()
        rag_mock.aquery.return_value = _make_rag_response(query="What are the limitations of the Transformer architecture?")

        handler = self._make_handler(llm=llm_mock, rag=rag_mock, memory=memory_mock)
        result = await handler.handle(session_id="s1", query="What about its limitations?")

        assert result.is_follow_up is True
        assert result.original_query == "What about its limitations?"
        assert result.rewritten_query == "What are the limitations of the Transformer architecture?"
        # RAG should be called with the rewritten query
        rag_mock.aquery.assert_called_once()
        call_kwargs = rag_mock.aquery.call_args
        assert call_kwargs[0][0] == "What are the limitations of the Transformer architecture?"

    @pytest.mark.asyncio
    async def test_handle_stores_messages(self):
        """User and assistant messages stored in ConversationMemory."""
        memory_mock = AsyncMock()
        memory_mock.get_history.return_value = []

        rag_mock = AsyncMock()
        rag_mock.aquery.return_value = _make_rag_response(answer="Transformers are great [1].")

        handler = self._make_handler(rag=rag_mock, memory=memory_mock)
        await handler.handle(session_id="s1", query="Explain transformers")

        # Should have called add_message twice: user + assistant
        assert memory_mock.add_message.call_count == 2
        user_call = memory_mock.add_message.call_args_list[0]
        assert user_call[1]["role"] == "user" or user_call[0][1] == "user"
        assistant_call = memory_mock.add_message.call_args_list[1]
        assert assistant_call[1]["role"] == "assistant" or assistant_call[0][1] == "assistant"

    @pytest.mark.asyncio
    async def test_handle_no_memory(self):
        """Works without ConversationMemory (memory=None)."""
        rag_mock = AsyncMock()
        rag_mock.aquery.return_value = _make_rag_response()

        handler = self._make_handler(rag=rag_mock, memory=None)
        result = await handler.handle(session_id="s1", query="What about its limitations?")

        # Should still work, treating as standalone (no history)
        assert result.is_follow_up is False
        assert result.response is not None

    @pytest.mark.asyncio
    async def test_handle_passes_rag_params(self):
        """top_k, categories, temperature forwarded to RAGChain."""
        rag_mock = AsyncMock()
        rag_mock.aquery.return_value = _make_rag_response()

        handler = self._make_handler(rag=rag_mock)
        await handler.handle(
            session_id="s1",
            query="Explain transformers",
            top_k=5,
            categories=["cs.AI"],
            temperature=0.3,
        )

        rag_mock.aquery.assert_called_once()
        call_kwargs = rag_mock.aquery.call_args[1]
        assert call_kwargs["top_k"] == 5
        assert call_kwargs["categories"] == ["cs.AI"]
        assert call_kwargs["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_handle_rag_error_propagates(self):
        """RAG errors are NOT swallowed."""
        rag_mock = AsyncMock()
        rag_mock.aquery.side_effect = RuntimeError("RAG failed")

        handler = self._make_handler(rag=rag_mock)
        with pytest.raises(RuntimeError, match="RAG failed"):
            await handler.handle(session_id="s1", query="Explain transformers")

    @pytest.mark.asyncio
    async def test_handle_stream_returns_iterator(self):
        """Streaming mode returns async iterator of tokens."""
        rag_mock = AsyncMock()
        rag_mock.aquery_stream.return_value = _mock_stream("Hello ", "world")

        handler = self._make_handler(rag=rag_mock)
        stream = await handler.handle_stream(session_id="s1", query="Explain transformers")

        tokens = []
        async for token in stream:
            tokens.append(token)

        assert "Hello " in tokens
        assert "world" in tokens

    @pytest.mark.asyncio
    async def test_handle_stream_follow_up(self):
        """Streaming follow-up rewrites query before streaming."""
        memory_mock = AsyncMock()
        memory_mock.get_history.return_value = _make_history([("Explain GANs", "GANs are generative adversarial networks...")])

        llm_mock = AsyncMock()
        llm_mock.generate.return_value = MagicMock(text="What are the limitations of GANs?")

        rag_mock = AsyncMock()
        rag_mock.aquery_stream.return_value = _mock_stream("GANs ", "have ", "mode collapse")

        handler = self._make_handler(llm=llm_mock, rag=rag_mock, memory=memory_mock)
        stream = await handler.handle_stream(session_id="s1", query="What about its limitations?")

        tokens = []
        async for token in stream:
            tokens.append(token)

        assert len(tokens) > 0
        # RAG stream should have been called with the rewritten query
        rag_mock.aquery_stream.assert_called_once()
        call_args = rag_mock.aquery_stream.call_args
        assert call_args[0][0] == "What are the limitations of GANs?"

    @pytest.mark.asyncio
    async def test_handle_stream_stores_messages_after_complete(self):
        """User + assistant messages stored after stream is consumed."""
        memory_mock = AsyncMock()
        memory_mock.get_history.return_value = []

        rag_mock = AsyncMock()
        rag_mock.aquery_stream.return_value = _mock_stream("Hello ", "world")

        handler = self._make_handler(rag=rag_mock, memory=memory_mock)
        stream = await handler.handle_stream(session_id="s1", query="Explain transformers")

        # Messages should NOT be stored yet (stream not consumed)
        # Consume stream
        tokens = []
        async for token in stream:
            tokens.append(token)

        # Now messages should be stored
        assert memory_mock.add_message.call_count == 2

    @pytest.mark.asyncio
    async def test_handle_stream_no_memory(self):
        """Streaming works without memory."""
        rag_mock = AsyncMock()
        rag_mock.aquery_stream.return_value = _mock_stream("test")

        handler = self._make_handler(rag=rag_mock, memory=None)
        stream = await handler.handle_stream(session_id="s1", query="test query")

        tokens = []
        async for token in stream:
            tokens.append(token)

        assert tokens == ["test"]
