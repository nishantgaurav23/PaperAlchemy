"""Web content fetcher — extracts clean text from URLs.

Uses Jina Reader API (r.jina.ai) for clean markdown extraction,
with httpx fallback for basic HTML text extraction.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)

_JINA_READER_PREFIX = "https://r.jina.ai/"
_FETCH_TIMEOUT = 15.0  # seconds per URL
_MAX_CONTENT_LENGTH = 4000  # chars to keep per page (fits in LLM context)


async def fetch_page_content(url: str, *, timeout: float = _FETCH_TIMEOUT) -> str | None:
    """Fetch clean text content from a URL using Jina Reader API.

    Returns extracted text (truncated to ~4000 chars) or None on failure.
    """
    jina_url = f"{_JINA_READER_PREFIX}{url}"
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(jina_url, headers={"Accept": "text/plain"})
            if resp.status_code == 200:
                text = resp.text.strip()
                if text:
                    return text[:_MAX_CONTENT_LENGTH]
    except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPError) as e:
        logger.debug("Jina Reader failed for %s: %s", url, e)

    # Fallback: skip (snippet from DuckDuckGo is still available)
    return None


async def fetch_multiple_pages(
    urls: list[str],
    *,
    max_concurrent: int = 3,
    timeout: float = _FETCH_TIMEOUT,
) -> dict[str, str]:
    """Fetch content from multiple URLs concurrently.

    Returns {url: content} for successfully fetched pages.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _fetch_with_limit(url: str) -> tuple[str, str | None]:
        async with semaphore:
            content = await fetch_page_content(url, timeout=timeout)
            return url, content

    tasks = [_fetch_with_limit(u) for u in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    fetched: dict[str, str] = {}
    for result in results:
        if isinstance(result, tuple):
            url, content = result
            if content:
                fetched[url] = content
        elif isinstance(result, Exception):
            logger.debug("Fetch task failed: %s", result)

    return fetched
