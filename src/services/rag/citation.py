"""Citation enforcement for RAG responses (S5.5).

Post-processes LLM output to parse, validate, and format citations.
Ensures every response has proper inline [N] references mapped to
real papers with title, authors, and arXiv links.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from src.services.rag.models import RAGResponse, SourceReference

# Regex: match [N] where N is a positive integer (1+).
# Handles nested brackets [[N]] by matching the inner [N].
_CITATION_RE = re.compile(r"\[(\d+)\]")

# Patterns to detect LLM-generated source sections to strip.
_SOURCES_SECTION_RE = re.compile(
    r"\n*\s*\*{0,2}Sources:?\*{0,2}\s*\n(?:.*\n?)*$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class CitationValidation:
    """Result of validating inline citations against the source list."""

    valid_citations: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    uncited_sources: list[int] = field(default_factory=list)
    citation_coverage: float = 0.0
    is_valid: bool = False


@dataclass
class CitationResult:
    """Result of citation enforcement on a RAG response."""

    formatted_answer: str = ""
    validation: CitationValidation = field(default_factory=CitationValidation)
    sources_markdown: str = ""


# ---------------------------------------------------------------------------
# FR-1: Citation Parser
# ---------------------------------------------------------------------------


def parse_citations(text: str) -> list[int]:
    """Extract sorted, deduplicated positive-integer citation indices from text.

    Matches [1], [2], etc. Ignores [0], [-1], [abc], [1-3].
    """
    indices: set[int] = set()
    for match in _CITATION_RE.finditer(text):
        raw = match.group(1)
        # Reject if it looks like a range (check char before digit group)
        start = match.start()
        if start > 0 and text[start - 1] == "[":
            # Nested bracket like [[1]] — still valid, extract inner
            pass
        val = int(raw)
        if val >= 1:
            indices.add(val)
    return sorted(indices)


# ---------------------------------------------------------------------------
# FR-2: Citation Validator
# ---------------------------------------------------------------------------


def validate_citations(
    cited_indices: list[int],
    sources: list[SourceReference],
) -> CitationValidation:
    """Validate that cited indices map to actual sources."""
    source_indices = {s.index for s in sources}

    valid = sorted(i for i in cited_indices if i in source_indices)
    invalid = sorted(i for i in cited_indices if i not in source_indices)
    cited_set = set(cited_indices)
    uncited = sorted(i for i in source_indices if i not in cited_set)

    n_sources = len(sources)
    coverage = len(valid) / n_sources if n_sources > 0 else 0.0

    # is_valid: at least one valid citation AND no invalid citations
    # Special case: both empty → valid (nothing to cite)
    if not cited_indices and not sources:
        is_valid = True
    elif not cited_indices:
        is_valid = False
    else:
        is_valid = len(valid) > 0 and len(invalid) == 0

    return CitationValidation(
        valid_citations=valid,
        invalid_citations=invalid,
        uncited_sources=uncited,
        citation_coverage=coverage,
        is_valid=is_valid,
    )


# ---------------------------------------------------------------------------
# FR-3: Source List Formatter
# ---------------------------------------------------------------------------


def _extract_year(arxiv_id: str) -> str:
    """Extract year from arxiv_id (e.g., '2301.00001' → '2023')."""
    if arxiv_id and len(arxiv_id) >= 2:
        prefix = arxiv_id[:2]
        try:
            yy = int(prefix)
            return f"20{yy:02d}"
        except ValueError:
            pass
    return ""


def _format_authors(authors: list[str]) -> str:
    """Format author list: ≤3 listed, >3 uses 'et al.'."""
    if not authors:
        return ""
    if len(authors) <= 3:
        return ", ".join(authors)
    return f"{authors[0]} et al."


def format_source_list(
    sources: list[SourceReference],
    *,
    cited_indices: list[int] | None = None,
) -> str:
    """Generate standardized markdown source list.

    Args:
        sources: List of source references.
        cited_indices: If provided, only include sources whose index is in this list.
    """
    if not sources:
        return ""

    if cited_indices is not None:
        cited_set = set(cited_indices)
        sources = [s for s in sources if s.index in cited_set]
        if not sources:
            return ""

    lines = ["**Sources:**"]
    for source in sources:
        title = source.title or source.arxiv_id or "Unknown"
        year = _extract_year(source.arxiv_id) if source.arxiv_id else ""
        authors_str = _format_authors(source.authors)

        # Pick the best URL: arxiv_url for papers, generic url for web
        link_url = source.arxiv_url or source.url or ""
        title_part = f"[{title}]({link_url})" if link_url else title

        # Add source type badge
        badge = ""
        if source.source_type == "arxiv":
            badge = " `[arXiv]`"
        elif source.source_type == "web":
            badge = " `[Web]`"

        parts = [f"{source.index}. {title_part}{badge}"]
        if authors_str:
            parts.append(f" \u2014 {authors_str}")
        if year:
            parts.append(f", {year}")

        lines.append("".join(parts))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# FR-4: Citation Enforcer
# ---------------------------------------------------------------------------


def _strip_sources_section(text: str) -> str:
    """Remove any LLM-generated Sources section from the answer text."""
    return _SOURCES_SECTION_RE.sub("", text).rstrip()


def enforce_citations(response: RAGResponse) -> CitationResult:
    """Post-process a RAGResponse: parse, validate, format, append ALL sources."""
    answer = response.answer.strip()
    sources = response.sources

    if not answer and not sources:
        return CitationResult()

    # 1. Parse citations from answer
    cited_indices = parse_citations(answer)

    # 2. Validate
    validation = validate_citations(cited_indices, sources)

    # 3. Strip any existing sources section the LLM may have generated.
    #    The frontend renders SourceCards separately from message.sources,
    #    so we must NOT include a markdown sources list in the answer text
    #    (otherwise it shows up twice: once as markdown, once as SourceCards).
    cleaned = _strip_sources_section(answer)

    # 4. Format source list — kept in sources_markdown for API consumers
    #    that don't render SourceCards (e.g., Telegram bot, CLI)
    sources_md = format_source_list(sources) if sources else ""

    # 5. Compose formatted answer: answer text only (no appended sources).
    #    Downstream consumers that need sources in text form can use sources_markdown.
    formatted = cleaned

    return CitationResult(
        formatted_answer=formatted,
        validation=validation,
        sources_markdown=sources_md,
    )


# ---------------------------------------------------------------------------
# FR-5: Streaming Citation Support
# ---------------------------------------------------------------------------


class _CitationStream:
    """Async iterator that streams tokens through in real-time, strips
    LLM-generated source sections, and appends a standardized source list.

    Tokens are forwarded as they arrive.  A trailing buffer is kept so that
    an LLM-generated "Sources:" block at the end can be detected and stripped
    without delaying the main body of the response.
    """

    # We buffer the last N characters to detect "Sources:" sections that the
    # LLM may append at the end.  Only the buffer is subject to stripping.
    _BUFFER_TRIGGER = "\nSources"  # shortest prefix that signals a sources block

    def __init__(self, tokens: AsyncIterator[str], sources: list[SourceReference]) -> None:
        self._tokens = tokens
        self._sources = sources
        self._accumulated = ""
        self.validation: CitationValidation | None = None
        self._done = False

    def __aiter__(self):
        return self._stream()

    async def _stream(self) -> AsyncIterator[str]:
        """Stream tokens, buffering the tail to detect and strip LLM-generated Sources sections."""
        buffer = ""

        async for token in self._tokens:
            self._accumulated += token
            buffer += token

            # Check if the buffer might contain the start of a Sources section.
            # Hold back text once we see a potential trigger so we can strip it.
            trigger_pos = buffer.find(self._BUFFER_TRIGGER)
            if trigger_pos >= 0:
                # Yield everything before the trigger
                if trigger_pos > 0:
                    yield buffer[:trigger_pos]
                # Hold the rest in the buffer — it might be a Sources block
                buffer = buffer[trigger_pos:]
            elif len(buffer) > len(self._BUFFER_TRIGGER) + 50:
                # Safe to flush most of the buffer (keep tail for trigger detection)
                flush_to = len(buffer) - len(self._BUFFER_TRIGGER)
                yield buffer[:flush_to]
                buffer = buffer[flush_to:]

        # Stream ended — check if the buffer is a Sources section to strip
        if buffer:
            cleaned_buffer = _strip_sources_section(buffer)
            if cleaned_buffer.strip():
                yield cleaned_buffer

        if not self._accumulated:
            self._done = True
            return

        # Compute validation on the full accumulated text (after stripping sources)
        cleaned = _strip_sources_section(self._accumulated)
        cited = parse_citations(cleaned)
        self.validation = validate_citations(cited, self._sources)
        self._done = True


def stream_with_citations(
    tokens: AsyncIterator[str],
    sources: list[SourceReference],
) -> _CitationStream:
    """Wrap a token stream to append citation enforcement at the end.

    Returns a _CitationStream that can be iterated and has a .validation
    property available after the stream is consumed.
    """
    return _CitationStream(tokens, sources)
